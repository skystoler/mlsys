import pickle as pkl

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import tvm

# import tvm.testing
from chapter2 import class_names, download_model_by_wget, load_data
from matplotlib import pyplot as plt
from torch import nn
from torchvision import transforms
from tvm import relax, te, topi
from tvm.script import tir as T


def pytorch_model():
    list = []
    list.append(
        nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), bias=True)
    )
    list.append(nn.ReLU())
    list.append(nn.MaxPool2d(kernel_size=(2, 2)))
    list.append(nn.Flatten())
    list.append(nn.Linear(in_features=5408, out_features=100, bias=True))
    list.append(nn.ReLU())
    list.append(nn.Linear(in_features=100, out_features=10, bias=True))
    list.append(nn.Softmax(dim=1))

    # Load the weight map from file.
    # The prediction accuracy of the weight map on test data is around 83.3%.
    weight_map = pkl.load(open("fasionmnist_mlp_assignment_params.pkl", "rb"))
    model = nn.Sequential(*list).cpu()
    name_map = {
        "0.weight": "conv2d_weight",
        "0.bias": "conv2d_bias",
        "4.weight": "linear0_weight",
        "4.bias": "linear0_bias",
        "6.weight": "linear1_weight",
        "6.bias": "linear1_bias",
    }
    for name, param in model.named_parameters():
        param.data = torch.from_numpy(weight_map[name_map[name]]).cpu()
    return model


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        print_img = True
        for data, label in test_loader:
            data, label = data.cpu(), label.cpu()
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, label, reduction="sum").item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            if print_img:
                imshow(data[0])
                print(
                    "predict: {}, label: {}".format(
                        class_names[pred[0][0]], class_names[label[0]]
                    )
                )
                print_img = False
            correct += pred.eq(label.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# Here we use topi to difine pytorch layer
def conv2d_1(X, weight, bias):
    Y = tvm.topi.nn.conv2d(X, weight, 1, 0, 1)
    return tvm.topi.add(Y, bias)


def relu_2(X):
    return tvm.topi.nn.relu(X)


def maxPool_3(X):
    return tvm.topi.nn.pool2d(
        data=X,
        kernel=[2, 2],
        stride=[2, 2],
        dilation=[1, 1],
        padding=[0, 0, 0, 0, 0, 0],
        pool_type="max",
    )


def flatten_4(X):
    return tvm.topi.nn.flatten(X)


def linear_5(X, weight, bias):
    Y = tvm.topi.nn.dense(X, weight)
    return tvm.topi.add(Y, bias)


def relu_6(X):
    return tvm.topi.nn.relu(X)


def linear_7(X, weight, bias):
    Y = tvm.topi.nn.dense(X, weight)
    return tvm.topi.add(Y, bias)


def softMax_8(X):
    return tvm.topi.nn.softmax(X)


# Here we register a pytorch conv2d to use in emit_te like lv1_0
@tvm.register_func("env.conv2d", override=True)
def torch_conv2d(x: tvm.nd.NDArray, w: tvm.nd.NDArray, out: tvm.nd.NDArray):
    x_torch = torch.from_dlpack(x)
    w_torch = torch.from_dlpack(w)
    out_torch = torch.from_dlpack(out)
    t = torch.nn.functional.conv2d(x_torch, w_torch)
    out_torch.copy_(t)


def create_model_via_emit_te(batch_size, input_shape, weight_map):
    bb = relax.BlockBuilder()

    x = relax.Var("x", input_shape, relax.DynTensorType(batch_size, "float32"))

    conv2d_weight = relax.const(weight_map["conv2d_weight"], "float32")
    conv2d_bias = relax.const(weight_map["conv2d_bias"].reshape(1, 32, 1, 1), "float32")
    linear0_weight = relax.const(weight_map["linear0_weight"], "float32")
    linear0_bias = relax.const(weight_map["linear0_bias"].reshape(1, 100), "float32")
    linear1_weight = relax.const(weight_map["linear1_weight"], "float32")
    linear1_bias = relax.const(weight_map["linear1_bias"].reshape(1, 10), "float32")

    with bb.function("main", [x]):
        with bb.dataflow():
            lv1 = bb.emit_te(conv2d_1, x, conv2d_weight, conv2d_bias)
            # lv1_0 = bb.emit_te(
            #     relax.op.call_tir(relax.extern("env.conv2d")),
            #     (x, conv2d_weight),
            #     (4, 32, 26, 26),
            #     dtype="float32",
            # )
            lv2 = bb.emit_te(relu_2, lv1)
            lv3 = bb.emit_te(maxPool_3, lv2)
            lv4 = bb.emit_te(flatten_4, lv3)
            lv5 = bb.emit_te(linear_5, lv4, linear0_weight, linear0_bias)
            lv6 = bb.emit_te(relu_6, lv5)
            lv7 = bb.emit_te(linear_7, lv6, linear1_weight, linear1_bias)
            lv8 = bb.emit_te(softMax_8, lv7)

            gv = bb.emit_output(lv8)
        bb.emit_func_output(gv)

    return bb.get()


def build_mod(mod):
    exec = relax.vm.build(mod, "llvm")
    dev = tvm.cpu()
    vm = relax.VirtualMachine(exec, dev)
    return vm


def check_equivalence(mod, torch_model, test_loader):
    torch_model.eval()
    with torch.no_grad():
        rt_mod = build_mod(mod)
        for data, label in test_loader:
            data, label = data.cpu(), label.cpu()
            output_from_pytorch = torch_model(data)
            output_from_relax = rt_mod["main"](tvm.nd.array(data, tvm.cpu()))
            torch.allclose(output_from_pytorch, output_from_relax, rtol=1e-4)


def prepare_pytorch_model():
    test_loader = load_data()

    url = (
        "https://github.com/mlc-ai/web-data/raw/main/models/fasionmnist_mlp_params.pkl"
    )
    download_model_by_wget(url)

    test(pytorch_model(), test_loader)


def transform_pytorch_model_to_tvm(batch_size, input_shape, weight_map):
    test_data = torchvision.datasets.FashionMNIST(
        "./data",
        download=True,
        train=False,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False
    )

    mod = create_model_via_emit_te(batch_size, input_shape, weight_map)
    torch_model = pytorch_model()

    check_equivalence(mod, torch_model, test_loader)


def schedule_mod(batch_size, input_shape, weight_map):
    mod = create_model_via_emit_te(batch_size, input_shape, weight_map)
    sch = tvm.tir.Schedule(mod)

    # Step 1. Get blocks
    pad_temp = sch.get_block("pad_temp", "conv2d_1")

    # Step 2. Inline the padding block (if exists)
    sch.compute_inline(pad_temp)

    # Step 3. Get loops
    conv = sch.get_block("conv2d_nchw", "conv2d_1")

    # Step 4. Organize the loops
    i0, i1, i2, i3, i4, i5, i6 = sch.get_loops(conv)

    i0_0, i0_1 = sch.split(i0, factors=[2, 2])
    i1_0, i1_1 = sch.split(i1, factors=[None, 4])
    i2_0, i2_1 = sch.split(i2, factors=[None, 2])
    i3_0, i3_1 = sch.split(i3, factors=[None, 2])
    sch.reorder(i0_0, i1_0, i2_0, i3_0, i4, i5, i6, i0_1, i1_1, i2_1, i3_1)

    i0_0, i1_0, i2_0, i3_0, i4, i5, i6, i0_1, i1_1, i2_1, i3_1 = sch.get_loops(conv)

    sch.fuse(i0_0, i1_0, i2_0, i3_0)
    i0_0_i1_0_i2_0_i3_0_fuse, i4, i5, i6, i0_1, i1_1, i2_1, i3_1 = sch.get_loops(conv)

    sch.parallel(i0_0_i1_0_i2_0_i3_0_fuse)

    sch.fuse(i0_1, i1_1)
    sch.fuse(i2_1, i3_1)

    (
        i0_0_i1_0_i2_0_i3_0_fuse,
        i4,
        i5,
        i6,
        i0_1_i1_1_fused,
        i2_1_i3_1_fused,
    ) = sch.get_loops(conv)
    sch.unroll(i0_1_i1_1_fused)
    sch.vectorize(i2_1_i3_1_fused)

    sch.decompose_reduction(conv, i4)

    print(sch.mod.script())
    torch_model = pytorch_model()
    test_data = torchvision.datasets.FashionMNIST(
        "./data",
        download=True,
        train=False,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False
    )
    check_equivalence(sch.mod, torch_model, test_loader)


if __name__ == "__main__":
    batch_size = 4
    input_shape = (batch_size, 1, 128, 128)
    prepare_pytorch_model()

    weight_map = pkl.load(open("fasionmnist_mlp_assignment_params.pkl", "rb"))

    transform_pytorch_model_to_tvm(batch_size, input_shape, weight_map)
    # schedule_mod(batch_size, input_shape, weight_map)
