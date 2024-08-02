from typing import List
import torch

import torch_tensorrt.fx.tracer.acc_tracer.acc_tracer as acc_tracer
import deeplearning.trt.fx2trt.converter.converters
from torch.fx.experimental.fx2trt.fx2trt import InputTensorSpec, TRTInterpreter
from torch_tensorrt.fx import TRTModule

# import deeplearning.trt as trt
import nn
import torch.fx

"""
Torch dynamo is torch 2.0,
which captures compute graph as a fx graph 
by interpreting python bytecode

Torch dynamo compile before excuting, it is a JIT compiler,
it can use tensorrt as backend.
"""

"""
1. Capture compute graph
"""
# If meeting unsupported op, torch dynamo create graph break, 
# divide compute graph into subgraphs,
# return unsupported op to python interpreter 

# Torch dynamo is static shape mode
# Loops will be upunolled in dynamo
def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print(">>> my_compiler() invoked:")
    print(">>> FX graph:")
    gm.graph.print_tabular()
    print(f">>> Code:\n{gm.code}")
    return gm.forward  # return a python callable

@torch.compile(backend=my_compiler)
def foo(x, y):
    return (x + y) * x


"""
2. Convert a pytorch model to tensorrt engine 
"""
# Inputs have to be a List of Tensors
# don not use dynamic control flow
def torch2trt(model_url, inputs):
    # step 1: Trace the model with acc_tracer
    # acc_tracer helps to convert didderent opearators to acc ops
    # like torch.add, builtin.add and torch.Tensor.add
    model = torch.load(model_url)
    acc_mod = acc_tracer.trace(model, inputs)
    
    # use explicit batch mode, supporting dynamic shape and loops
    # 1. Shapes of inputs, outputs and activations are fixed except batch dimension. 
    # 2. Inputs, outputs and activations have batch dimension as the major dimension. 
    # 3.  All the operators in the model do not modify batch dimension (permute, transpose, split, etc.) 
    # or compute over batch dimension (sum, softmax, etc.).

    # # Currently we only support one set of dynamic range. 
    # User may set other dimensions but it is not promised to work for any models
    input_specs = [
        InputTensorSpec(
            shape = (-1, 2, 3),
            dtype = torch.float32,
            # shape_ranges: (min_shape, optimize_target_shape, max_shape)
            shape_ranges = [
                ((1, 2, 3), (4, 2, 3), (100, 2, 3)),
            ],
        ),
        InputTensorSpec(
            shape=(1, 4, 5), 
            dtype=torch.float32
        ),
    ]

    interpreter = TRTInterpreter(
        acc_mod, input_specs, explicit_batch_dimension=True
    )

    # RuntimeError: Conversion of function xxx not currently supported! 
    # This means we don’t have the support for this xxx operator. 
    trt_interpreter_result = interpreter.run(
        max_batch_size=64,
        max_workspace_size=1 << 25,
        sparse_weights=False,
        force_fp32_output=False,
        strict_type_constraints=False,
        algorithm_selector=None,
        timing_cache=None,
        profiling_verbosity=None,
    )
    
    trt_model = TRTModule(
        trt_interpreter_result.engine,
        trt_interpreter_result.input_names,
        trt_interpreter_result.output_names
    )

    return trt_model


"""
3. FX2TRT
"""
def fx2trt(network):
    """
    step 1: define a new acc op
    """
    # all acc ops should only take kwargs as inputs, 
    # therefore we need the "*" at the beginning.
    @register_acc_op
    def foo(*, input, other, alpha=1.0):
        return input + alpha * other
    
    """
    step2: register a mapping
    """
    this_arg_is_optional = True
    @register_acc_op_mapping(
        op_and_target=("call_function", torch.add),
        arg_replacement_tuples=[
            ("input", "input"),
            ("other", "other"),
            ("alpha", "alpha", this_arg_is_optional),
        ],
    )
    # This one is designed to reduce the redundant op registration 
    # can use a function to map to one or more existing acc ops 
    # throught some combinations. 

    # @register_custom_acc_mapper_fn(
    #     op_and_target=("call_function", torch.add),
    #     arg_replacement_tuples=[
    #         ("input", "input"),
    #         ("other", "other"),
    #         ("alpha", "alpha", this_arg_is_optional),
    #     ],
    # )
    def custom_mapper(node: torch.fx.Node, _: nn.Module) -> torch.fx.Node:
        """
        `node` is original node, which is a call_function node with target
        being torch.add.
        """
        alpha = 1
        if "alpha" in node.kwargs:
            alpha = node.kwargs["alpha"]
        foo_kwargs = {"input": node["input"], "other": node["other"], "alpha": alpha}
        with node.graph.inserting_before(node):
            foo_node = node.graph.call_function(foo, kwargs=foo_kwargs)
            foo_node.meta = node.meta.copy()
            return foo_node
    
    """
    step 3: add a new converter to map the acc ops to a TensorRT layer.
    """
    @tensorrt_converter(acc_ops.sigmoid)
    def acc_ops_sigmoid(network, target, args, kwargs, name):
        # network: TensorRT network. We'll be adding layers to it.
        # The rest arguments are attributes of fx node.
        input_val = kwargs['input']
        if not isinstance(input_val, trt.tensorrt.ITensor):
            raise RuntimeError(f'Sigmoid received input {input_val} that is not part '
                            'of the TensorRT region!')

        layer = network.add_activation(input=input_val, type=trt.ActivationType.SIGMOID)
        layer.name = name
        return layer.get_output(0)
    
    #(to do) how to use
    custom_mapper()
    return acc_ops_sigmoid(network)

"""
try capture compute graph
"""
if __name__ == '__main__':
    # 1.
    a = torch.randn((3, 2), dtype=torch.float32)
    b = torch.randn((3, 2), dtype=torch.float32)
    foo(a, b)

    # 2.
    inputs = [
        torch.rand((1,2,3), dtype=torch.float32),
        torch.rand((1,4,5), dtype=torch.float32),
    ]
    model_url = " "

    # convert
    trt_model = torch2trt(model_url, inputs)

    # use
    outputs = trt_model(*inputs)

    # # save
    # torch.save(trt_model, "trt.pt")

    # # reload
    # reload_trt_model = torch.load("trt.pt")

    # 3.
    trt_model_from_fx = fx2trt(*inputs)