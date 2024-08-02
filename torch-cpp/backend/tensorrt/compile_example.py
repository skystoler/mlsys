import torch
import torch_tensorrt
import torchvision.model as models

"""
Tensor RT in torch-cpp
Use of tensorrt is bind to dynamo frontend
Compilation with torch_tensorrt.compile / torch.compile

[docs]def compile(
    module: Any,
    ir: str = "default",
    inputs: Optional[Sequence[Input | torch.Tensor | InputTensorSpec]] = None,
    enabled_precisions: Optional[Set[torch.dtype | dtype]] = None,
    **kwargs: Any,
) -> (
    torch.nn.Module | torch.jit.ScriptModule | torch.fx.GraphModule | Callable[..., Any]
):
    Compile a PyTorch module for NVIDIA GPUs using TensorRT

    Takes a existing PyTorch module and a set of settings to configure the compiler
    and using the path specified in ``ir`` lower and compile the module to TensorRT
    returning a PyTorch Module back

    Converts specifically the forward method of a Module

    Arguments:
        module (Union(torch.nn.Module,torch.jit.ScriptModule): Source module

    Keyword Arguments:
        inputs (List[Union(torch_tensorrt.Input, torch.Tensor)]): **Required** List of specifications of input shape, dtype and memory layout for inputs to the module. This argument is required. Input Sizes can be specified as torch sizes, tuples or lists. dtypes can be specified using
            torch datatypes or torch_tensorrt datatypes and you can use either torch devices or the torch_tensorrt device type enum
            to select device type. ::

                inputs=[
                    torch_tensorrt.Input((1, 3, 224, 224)), # Static NCHW input shape for input #1
                    torch_tensorrt.Input(
                        min_shape=(1, 224, 224, 3),
                        opt_shape=(1, 512, 512, 3),
                        max_shape=(1, 1024, 1024, 3),
                        dtype=torch.int32
                        format=torch.channel_last
                    ), # Dynamic input shape for input #2
                    torch.randn((1, 3, 224, 244)) # Use an example tensor and let torch_tensorrt infer settings
                ]

        enabled_precision (Set(Union(torch.dtype, torch_tensorrt.dtype))): The set of datatypes that TensorRT can use when selecting kernels

        ir (str): The requested strategy to compile.
        ["torchscript/ts", "fx", "dynamo", "torch_compile"]
        (Options: default - Let Torch-TensorRT decide, ts - TorchScript with scripting path)
        if ir = "torch_compile", it is Equivalently as so:
            optimized_model = torch.compile(model, backend=”torch_tensorrt”, options={“enabled_precisions”: enabled_precisions, …}); optimized_model(*inputs)
        if ir = "torchscript" or "ts", 
            compiled_model = torchscript_compile()
        if ir = "fx",
            compiled_model = fx_compile()
        if ir = "dynamo",
            compiled_model = dynamo_compile()
        **kwargs: Additional settings for the specific requested strategy 
        See submodules for more info in https://pytorch.org/TensorRT/dynamo/torch_compile.html
    Returns:
        torch.nn.Module: Compiled Module, when run it will execute via TensorRT
    """

"""
1. ir = "torch_compile"
"""
model = models.resnet(pretrained=True).half().eval.to("cuda")
inputs = [torch.randn((1, 3, 224, 224), dtype=torch.fp32).to("cuda").half()]

# Enabled precision for TensorRT optimization
enabled_precisions = {torch.float}

# Whether to print verbose logs
debug = True

# Workspace size for TensorRT
workspace_size = 20 << 30

# Maximum number of TRT Engines
# (Lower value allows more graph segmentation)
min_block_size = 7

# Operations to Run in Torch, regardless of converter support
torch_executed_ops = {}

compiled_model_from_torchcompile = torch_tensorrt.compile(
    model,
    ir="torch_compile",
    inputs=input,
    enable_precisons=enabled_precisions,
    debug=debug,
    workspace_size=workspace_size,
    min_block_size=min_block_size,
    torch_executed_ops=torch_executed_ops,
)

"""
# equivalant to the above code

compilation_kwargs = {
    "enabled_precisions": enabled_precisions,
    "debug": debug,
    "workspace_size": workspace_size,
    "min_block_size": min_block_size,
    "torch_executed_ops": torch_executed_ops,
}

compiled_model_from_torchcompile = torch.compile(
    model,
    backend="torch_tensorrt",
    dynamic=False,
    options=compilation_kwargs,
)
"""

"""
2. ir = "dynamo", use cuda graph
ir="torch_compile" should work with cudagraphs as well.
"""
opt = torch_tensorrt.compile(
    model,
    ir="dynamo",
    inputs=torch_tensorrt.Input(
        min_shape=(1, 3, 224, 224),
        opt_shape=(8, 3, 224, 224),
        max_shape=(16, 3, 224, 224),
        dtype=torch.float,
        name="x",
    ),
)

"""
Inference
"""
if __name__ == "__main__":
    # Does not cause recompilation (same batch size as input)
    new_inputs = [torch.randn((1, 3, 224, 224)).half().to("cuda")]
    new_outputs = compiled_model_from_torchcompile(*new_inputs)

    # Does cause recompilation (new batch size)
    new_batch_size_inputs = [torch.randn((8, 3, 224, 224)).half().to("cuda")]
    new_batch_size_outputs = compiled_model_from_torchcompile(*new_batch_size_inputs)

    """
    Avoid recompilation by specifying dynamic shapes before Torch-TRT compilation
    """
    # The following code illustrates the workflow using ir=torch_compile (which uses torch.compile under the hood)
    inputs_bs8 = torch.randn((8, 3, 224, 224)).half().to("cuda")
    # This indicates dimension 0 of inputs_bs8 is dynamic whose range of values is [2, 16]
    torch._dynamo.mark_dynamic(inputs_bs8, 0, min=2, max=16)
    compiled_model_from_torchcompile_dynamicshape = torch_tensorrt.compile(
        model,
        ir="torch_compile",
        inputs=inputs_bs8,
        enabled_precisions=enabled_precisions,
        debug=debug,
        workspace_size=workspace_size,
        min_block_size=min_block_size,
        torch_executed_ops=torch_executed_ops,
    )
    outputs_bs8 = compiled_model_from_torchcompile_dynamicshape(inputs_bs8)

    # No recompilation happens for batch size = 12
    inputs_bs12 = torch.randn((12, 3, 224, 224)).half().to("cuda")
    outputs_bs12 = compiled_model_from_torchcompile_dynamicshape(inputs_bs12)

    """
    The following code illustrates the workflow using ir=dynamo (which uses torch.export APIs under the hood)
    dynamic shapes for any inputs are specified using torch_tensorrt.Input API
    """
    compile_spec = {
        "inputs": [
            torch_tensorrt.Input(
                min_shape=(1, 3, 224, 224),
                opt_shape=(8, 3, 224, 224),
                max_shape=(16, 3, 224, 224),
                dtype=torch.half,
            )
        ],
        "enabled_precisions": enabled_precisions,
        "ir": "dynamo",
    }
    trt_model = torch_tensorrt.compile(model, **compile_spec)

    # No recompilation happens for batch size = 12
    inputs_bs12 = torch.randn((12, 3, 224, 224)).half().to("cuda")
    outputs_bs12 = trt_model(inputs_bs12)

    """
    Inference using the Cudagraphs Integration
    """
    # We can enable the cudagraphs API with a context manager
    # If we provide new input shapes, cudagraphs will re-record the graph
    with torch_tensorrt.runtime.enable_cudagraphs():
        out_trt = opt(inputs)

    """
    # Alternatively, we can set the cudagraphs mode for the session
    torch_tensorrt.runtime.set_cudagraphs_mode(True)
    out_trt = opt(inputs)

    # We can also turn off cudagraphs mode and perform inference as normal
    torch_tensorrt.runtime.set_cudagraphs_mode(False)
    out_trt = opt(inputs)
    """

    # Finally, we use Torch utilities to clean up the workspace
    torch._dynamo.reset()