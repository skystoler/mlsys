# This program tells us the difference of numpy and tvm to
# calculate matrix multiply and relu
# Tvm represents tensor by TensorIR

import numpy as np
import tvm
from tvm.ir.module import IRModule
from tvm.script import tir as T

# low level calculated by numpy
a_np = np.random.rand(128, 128).astype("float32")
b_np = np.random.rand(128, 128).astype("float32")

c_mm_relu = np.maximum(a_np @ b_np, 0)


def lnumpy_mm_relu(A: np.ndarray, B: np.ndarray, C: np.ndarray):
    Y = np.empty((128, 128), dtype="float32")
    for i in range(128):
        for j in range(128):
            for k in range(128):
                if k == 0:
                    Y[i][j] = 0
                Y[i][j] += A[i][k] * B[k][j]

    for i in range(128):
        for j in range(128):
            C[i][j] = max(Y[i][j], 0)


# TensorIR realization
# This decorator means  thisc class is a IRModule
# for saving tensor def
@tvm.script.ir_module
class MyModule:
    @T.prim_func
    # Use buffer instead of ndarray to make use of shape and data type
    def mm_relu(
        A: T.Buffer((128, 128), "float32"),
        B: T.Buffer((128, 128), "float32"),
        C: T.Buffer((128, 128), "float32"),
    ):
        # Extra information about def
        T.func_attr({"global_symbol": "mm_relu", "tir.noalias": True})
        Y = T.allocate_buffer((128, 128), dtype="float32")

        # Grammar sugar for writing many loops
        for i, j, k in T.grid(128, 128, 128):
            # Extra structure
            with T.block("Y"):
                # Axis type: spatial and reduce

                # To provide extra information to
                # verify the correctness of the outer loops

                # Also does help to machine learning compile analysis
                # Such as parallel in spatial axis
                vi = T.axis.spatial(128, i)
                vj = T.axis.spatial(128, j)
                vk = T.axis.reduce(128, k)
                # We can also use belows to replace aboves
                # vi, vj, vk = T.axis.remap("SSR", [i, j, k])

                with T.init():
                    Y[vi, vj] = T.float32(0)
                Y[vi, vj] += A[vi, vk] * B[vk, vj]

        for i, j in T.grid(128, 128):
            with T.block("C"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = max(Y[vi, vj], T.float32(0))
