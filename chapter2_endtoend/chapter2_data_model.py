import os
import pickle as pkl
import tempfile

import matplotlib as plt
import numpy as np
import torch
import torchvision
import tvm
import wget
from tvm.ir.module import IRModule
from tvm.script import relax as R
from tvm.script import tir as T

class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


# Load data
def load_data():
    test_data = torchvision.datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)

    img, label = next(iter(test_loader))
    img = img.reshape(1, 28, 28).numpy()
    return img, label


def download_model_by_wget(url):
    tmpdir = tempfile.gettempdir()
    target_name = url.split("/")[-1]
    try:
        model_name = wget.download(url, out=os.path.join(tmpdir, target_name))
    except Exception:
        print("Download failed!!!")
    else:
        print(model_name + "download finish!")
        return target_name


def load_model_from_tmp(target_name):
    model = pkl.load(open("./tmp/" + target_name, "rb"))
    return model


# Paint figure
def paint(img, label):
    print("***************************")
    plt.figure()
    plt.imshow(img[0])
    plt.colorbar()
    plt.grid(False)
    plt.show()
    print("Class:", class_names[label[0]])
    print("**************************")


# Construct model as IRModule
@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def relu0(x: T.handle, y: T.handle):
        n = T.int64()
        X = T.match_buffer(x, (1, n), "float32")
        Y = T.match_buffer(y, (1, n), "float32")
        for i, j in T.grid(1, n):
            with T.block("Y"):
                vi, vj = T.axis.remap("SS", [i, j])
                Y[vi, vj] = T.max(X[vi, vj], T.float32(0))

    @T.prim_func
    def linear0(x: T.handle, w: T.handle, b: T.handle, z: T.handle):
        m, n, k = T.int64(), T.int64(), T.int64()
        X = T.match_buffer(x, (1, m), "float32")
        W = T.match_buffer(w, (n, m), "float32")
        B = T.match_buffer(b, (n,), "float32")
        Z = T.match_buffer(z, (1, n), "float32")
        Y = T.alloc_buffer((1, n), "float32")
        for i, j, k in T.grid(1, n, m):
            with T.block("Y"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    Y[vi, vj] = T.float32(0)
                Y[vi, vj] = Y[vi, vj] + X[vi, vk] * W[vj, vk]
        for i, j in T.grid(1, n):
            with T.block("Z"):
                vi, vj = T.axis.remap("SS", [i, j])
                Z[vi, vj] = Y[vi, vj] + B[vj]

    @R.function
    def main(
        x: R.Tensor((1, "m"), "float32"),
        w0: R.Tensor(("n", "m"), "float32"),
        b0: R.Tensor(("n",), "float32"),
        w1: R.Tensor(("k", "n"), "float32"),
        b1: R.Tensor(("k",), "float32"),
    ):
        m, n, k = T.int64(), T.int64(), T.int64()
        with R.dataflow():
            lv0 = R.call_dps_packed("linear0", (x, w0, b0), R.Tensor((1, n), "float32"))
            lv1 = R.call_dps_packed("relu0", (lv0,), R.Tensor((1, n), "float32"))
            out = R.call_dps_packed(
                "linear0", (lv1, w1, b1), R.Tensor((1, k), "float32")
            )
            R.output(out)
        return out


def lnumpy_linear0(X: np.ndarray, W: np.ndarray, B: np.ndarray, Z: np.ndarray):
    Y = np.empty((1, 128), dtype="float32")
    for i in range(1):
        for j in range(128):
            for k in range(784):
                if k == 0:
                    Y[i, j] = 0
                Y[i, j] = Y[i, j] + X[i, k] * W[j, k]

    for i in range(1):
        for j in range(128):
            Z[i, j] = Y[i, j] + B[j]


def lnumpy_relu0(X: np.ndarray, Y: np.ndarray):
    for i in range(1):
        for j in range(128):
            Y[i, j] = np.maximum(X[i, j], 0)


def lnumpy_linear1(X: np.ndarray, W: np.ndarray, B: np.ndarray, Z: np.ndarray):
    Y = np.empty((1, 10), dtype="float32")
    for i in range(1):
        for j in range(10):
            for k in range(128):
                if k == 0:
                    Y[i, j] = 0
                Y[i, j] = Y[i, j] + X[i, k] * W[j, k]

    for i in range(1):
        for j in range(10):
            Z[i, j] = Y[i, j] + B[j]


def lnumpy_mlp(data, w0, b0, w1, b1):
    lv0 = np.empty((1, 128), dtype="float32")
    lnumpy_linear0(data, w0, b0, lv0)

    lv1 = np.empty((1, 128), dtype="float32")
    lnumpy_relu0(lv0, lv1)

    out = np.empty((1, 10), dtype="float32")
    lnumpy_linear1(lv1, w1, b1, out)
    return out


def lnumpy_mlp_predict(target_name):
    mlp_params = load_model_from_tmp(target_name)
    result = lnumpy_mlp(
        img.reshape(1, 784),
        mlp_params["w0"],
        mlp_params["b0"],
        mlp_params["w1"],
        mlp_params["b1"],
    )

    pred_kind = result.argmax(axis=1)
    print("Low-level Numpy MLP Prediction:", class_names[pred_kind[0]])


# Create computation graph
def create_com_graph(target_name):
    def lnumpy_call_dps_packed(prim_func, inputs, shape, dtype):
        res = np.empty(shape, dtype=dtype)
        prim_func(*inputs, res)
        return res

    def lnumpy_mlp_with_call_dps_packed(data, w0, b0, w1, b1):
        lv0 = lnumpy_call_dps_packed(
            lnumpy_linear0, (data, w0, b0), (1, 128), dtype="float32"
        )
        lv1 = lnumpy_call_dps_packed(lnumpy_relu0, (lv0,), (1, 128), dtype="float32")
        out = lnumpy_call_dps_packed(
            lnumpy_linear1, (lv1, w1, b1), (1, 10), dtype="float32"
        )
        return out

    mlp_params = load_model_from_tmp(target_name)
    result = lnumpy_mlp_with_call_dps_packed(
        img.reshape(1, 784),
        mlp_params["w0"],
        mlp_params["b0"],
        mlp_params["w1"],
        mlp_params["b1"],
    )

    pred_kind = np.argmax(result, axis=1)
    print("Low-level Numpy with CallTIR Prediction:", class_names[pred_kind[0]])


# A program containing 2 dataflow blocks
"""
@R.function
def main(x: R.Tensor((1, "m"), "float32"),
        w0: R.Tensor(("n", "m"), "float32"),
        b0: R.Tensor(("n", ), "float32"),
        w1: R.Tensor(("k", "n"), "float32"),
        b1: R.Tensor(("k", ), "float32")):
    m, n, k = T.int64(), T.int64(), T.int64()

    with R.dataflow():
        lv0 = R.call_dps_packed("linear0", (x, w0, b0), R.Tensor((1, n), "float32"))
        gv0 = R.call_dps_packed("relu0", (lv0, ), R.Tensor((1, n), "float32"))
        R.output(gv0)

    with R.dataflow():
        out = R.call_dps_packed("linear0", (gv0, w1, b1), R.Tensor((1, k), "float32"))
        R.output(out)
    return out
"""

if __name__ == "__main__":
    img, label = load_data()
    paint(img, label)

    url = (
        "https://github.com/mlc-ai/web-data/raw/main/models/fasionmnist_mlp_params.pkl"
    )
    target_name = download_model_by_wget(url)

    lnumpy_mlp_predict(target_name)
    create_com_graph(target_name)
