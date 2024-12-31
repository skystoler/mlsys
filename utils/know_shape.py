import torch
from ocr.utils.predict_ops import load_script
import underframe.torch as uf_torch

# torch.classes.load_library('v4789cc_trt_executor_extension_py38.so')

 
# for p in model.parameters():
#     print(p.size())
def get_torchscript_input_shapes(model_name):
    device=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    tec = uf_torch.context.TorchExecutionContext(device=device, dtype=dtype)

    with tec.omni_context():
        model = load_script(model_name, freeze=True, optimize=True)
    print(dir(model._module.forward), list(model._module.forward.graph.inputs()))
    graph = model._module.graph
    inputs = list(graph.inputs())
    real_inputs = [inp for inp in inputs if not inp.debugName().startswith("self")]
    for inp in real_inputs:
        type_info = inp.type()
        # Check if the input is a tensor
        if type_info.kind() == 'TensorType':
            shape = type_info.sizes()
            if not shape:
                continue
            # Convert the shape to a list for readability
            shape_list = []
            for dim in shape:
                if dim.is_complete():
                    shape_list.append(dim.toCompleteInt())
                else:
                    shape_list.append('?')  # Unknown dimension

            print(f"Input '{inp.debugName()}' shape: {shape_list}")
        else:
            print(f"Input '{inp.debugName()}' is not a tensor")

# Usage
if __name__ == '__main__':
    #model_name = "http://version-control-test.oss-cn-hangzhou-zmf.aliyuncs.com/temp%2Fmodels%2Fdoc_beit%2Fframeless_table%2Fai_doc_table%2F1126%2Fcheckpoint-157-best_AiDocTable0823_ser_link_f1_score_98.08_tr.pth"
    model_name = "http://version-control-test.oss-cn-hangzhou-zmf.aliyuncs.com/temp%2Fmodels%2Fdoc_beit%2Fframeless_table%2Fai_doc_table%2F1126%2Fcheckpoint-157-best_AiDocTable0823_ser_link_f1_score_98.08.pth"
    get_torchscript_input_shapes(model_name)


    # answer: torch.Size([1, 3, 448, 448]) torch.Size([1, 2048]) torch.Size([1, 2048, 4]) torch.Size([1, 2048]) torch.Size([1, 2048])
    # input = torch.randn([1, 3, 448, 448]), torch.randn([1, 2048]), torch.randn([1, 2048, 4]), torch.randn([1, 2048]), torch.randn([1, 2048])
    # with tec.omni_context():
    #     model(*input)