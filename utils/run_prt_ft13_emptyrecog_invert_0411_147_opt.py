import torch
from ocr.utils.predict_ops import load_script
import underframe.torch as uf_torch

torch.classes.load_library('v4789cc_trt_executor_extension_py38.so')
model_name = "vc0c279_prt_ft13_emptyrecog_invert_0411_147_opt.ts"
device=torch.device("cuda:0")
dtype=torch.float16
tec = uf_torch.context.TorchExecutionContext(device=device, dtype=dtype)

with tec.omni_context():
    model = load_script(model_name, freeze=True, optimize=True)

iters = 10
warmup_iters = 5

#print_reco: torch.Size([42, 3, 64, 832]) torch.Size([42, 64, 832]) torch.Size([42, 1])
input = torch.randn([42, 3, 64, 832]), torch.randn([42, 64, 832]), torch.randn([42, 1])
for i in range(iters):
    if i == warmup_iters: 
        torch.cuda.cudart().cudaProfilerStart()

    if i >= warmup_iters: 
        torch.cuda.nvtx.range_push("iteration{}".format(i))
        torch.cuda.nvtx.range_push("forward")
        with tec.omni_context():
            output = model(*input)
        torch.cuda.nvtx.range_pop()
    
torch.cuda.cudart().cudaProfilerStop()