# This prgram introduces how to convert a toch fx graph into an IR module

import numpy as np
import torch
import torch.nn as nn
import tvm
from torch import fx
from torch.nn import functional as F
from tvm import relax, te
from tvm.ir.module import IRModule
from tvm.script import relax as R
from tvm.script import tir as T


def te_matmul(A: te.Tensor, B: te.Tensor) -> te.Tensor:
    assert A.shape[1] == B.shape[0]
    n = A.shape[0]
    m = B.shape[1]
    k = te.reduce_axis((0, A.shape[1]), name="k")
    return te.compute(
        (n, m), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="matmul"
    )


def te_relu(A: te.Tensor) -> te.Tensor:
    return te.compute(A.shape, lambda *i: te.max(A(*i), 0), name="relu")


def test_te():
    A = te.placeholder((128, 128), name="A", dtype="float32")
    B = te.placeholder((128, 128), name="B", dtype="float32")
    print(type(A))
    print(A.shape)
    C = te_matmul(A, B)
    te.create_prim_func([A, B, C]).show()

    X1 = te.placeholder((10,), name="X1", dtype="float32")
    Y1 = te_relu(X1)
    te.create_prim_func([X1, Y1]).show()

    X2 = te.placeholder((10, 20), name="X1", dtype="float32")
    Y2 = te_relu(X2)
    te.create_prim_func([X2, Y2]).show()

    D = te_relu(C)
    # We can pass without intermidiate value C
    te.create_prim_func([A, B, D]).show()
    # Of course we can pass all parmas
    te.create_prim_func([A, B, C, D]).show()


def test_blockbuilder():
    A = relax.Var("A", relax.TensorStructInfo((128, 128), "float32"))
    B = relax.Var("B", relax.TensorStructInfo((128, 128), "float32"))

    bb = relax.BlockBuilder()

    # Actually, emit_te does such things as belows:
    # 1.create placeholder for input A and B
    # 2.run them by te_matmul
    # 3.create a tensorIR func by create_prim_func
    # 4.call func by call_dps_packed
    with bb.function("main"):
        # Every emit call generates a variable inside a dataflow block
        with bb.dataflow():
            C = bb.emit_te(te_matmul, A, B)
            D = bb.emit_te(te_relu, C)
            E = bb.emit_output(D)
        # Specify params in the end
        bb.emit_func_output(E, params=[A, B])

    MyModule = bb.get()
    MyModule.show()


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.weight = nn.Parameter(torch.randn(128, 128))

    def forward(self, x):
        x = torch.matmul(x, self.weight)
        x = torch.relu(x)
        return x


# We constrcut mapping function as belows
def map_param(param: nn.Parameter):
    return relax.const(
        param.data.cpu().numpy(), relax.TensorStructInfo(param.data.shape, "float32")
    )


def fetch_attr(fx_mod, target: str):
    """Helper function to fetch an attr"""
    target_atoms = target.split(".")
    attr_itr = fx_mod
    for i, atom in enumerate(target_atoms):
        if not hasattr(attr_itr, atom):
            raise RuntimeError(
                f"Node referenced nonexistant target {'.'.join(target_atoms[:i])}"
            )
        attr_itr = getattr(attr_itr, atom)
    return attr_itr


def from_fx(fx_mod, input_shapes, call_function_map, call_module_map):
    input_index = 0
    # Create a nod_map to map fx.node to corresponding relax.var,
    # which represents translated node in IRModule
    node_map = {}
    named_modules = dict(fx_mod.named_modules())

    bb = relax.BlockBuilder()

    fn_inputs = []
    fn_output = None
    with bb.function("main"):
        with bb.dataflow():
            for node in fx_mod.graph.nodes:
                if node.op == "placeholder":
                    # create input placeholder
                    shape = input_shapes[input_index]
                    input_index += 1
                    input_var = relax.Var(
                        node.target, relax.TensorStructInfo(shape, "float32")
                    )
                    fn_inputs.append(input_var)
                    node_map[node] = input_var
                elif node.op == "get_attr":
                    node_map[node] = map_param(fetch_attr(fx_mod, node.target))
                elif node.op == "call_function":
                    node_map[node] = call_function_map[node.target](bb, node_map, node)
                elif node.op == "call_module":
                    named_module = named_modules[node.target]
                    node_map[node] = call_module_map[type(named_module)](
                        bb, node_map, node, named_module
                    )
                elif node.op == "output":
                    output = node_map[node.args[0]]
                    assert fn_output is None
                    fn_output = bb.emit_output(output)
        # output and finalize the function
        bb.emit_func_output(output, fn_inputs)
    return bb.get()


# Here we use fx graph in torch
def import_model_from_torch():
    model = MyModel()
    fx_module = fx.symbolic_trace(model)
    print(type(fx_module))
    # require tabulate module installed
    fx_module.graph.print_tabular()

    def map_matmul(bb, node_map, node: fx.Node):
        A = node_map[node.args[0]]
        B = node_map[node.args[1]]
        return bb.emit_te(te_matmul, A, B)

    def map_relu(bb, node_map, node: fx.Node):
        A = node_map[node.args[0]]
        return bb.emit_te(te_relu, A)

    MyModule = from_fx(
        fx_module,
        input_shapes=[(1, 128)],
        call_function_map={
            torch.matmul: map_matmul,
            torch.relu: map_relu,
        },
        call_module_map={},
    )

    MyModule.show()


if __name__ == "__main__":
    test_te()
    test_blockbuilder()
    import_model_from_torch()
