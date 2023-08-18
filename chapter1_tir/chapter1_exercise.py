import numpy as np
import tvm
from tvm.script import tir as T


@tvm.script.ir_module
class MyAdd:
    @T.prim_func
    def add(
        A: T.Buffer((4, 4), "int64"),
        B: T.Buffer((4,), "int64"),
        C: T.Buffer((4, 4), "int64"),
    ):
        T.func_attr({"global_symbol": "add", "tir.noalias": True})
        for i, j in T.grid(4, 4):
            with T.block("C"):
                vi = T.axis.spatial(4, i)
                vj = T.axis.spatial(4, j)
                C[vi, vj] = A[vi, vj] + B[vj]


@tvm.script.ir_module
class MyConv:
    @T.prim_func
    def conv(
        A: T.Buffer((1, 1, 8, 8), "int64"),
        W: T.Buffer((2, 1, 3, 3), "int64"),
        C: T.Buffer((1, 2, 6, 6), "int64"),
    ):
        T.func_attr({"global_symbol": "conv", "tir.noalias": True})
        for b, k, i, j, di, dj in T.grid(1, 2, 6, 6, 3, 3):
            with T.block("C"):
                vb, vk, vi, vj, vdi, vdj = T.axis.remap("SSSSRR", [b, k, i, j, di, dj])
                with T.init():
                    C[vb, vk, vi, vj] = T.int64(0)
                C[vb, vk, vi, vj] = C[vb, vk, vi, vj] + (
                    A[vb, vb, vi + vdi, vj + vdj] * W[vk, vb, vdi, vdj]
                )


@tvm.script.ir_module
class MyBmmRelu:
    @T.prim_func
    def bmm_relu(
        A: T.Buffer((16, 128, 128), "float32"),
        B: T.Buffer((16, 128, 128), "float32"),
        C: T.Buffer((16, 128, 128), "float32"),
    ):
        T.func_attr({"global_symbol": "bmm_relu", "tir.noalias": True})
        Y = T.alloc_buffer((16, 128, 128), dtype="float32")
        for b, i, j, k in T.grid(16, 128, 128, 128):
            with T.block("Y"):
                vb, vi, vj, vk = T.axis.remap("SSSR", [b, i, j, k])
                with T.init():
                    Y[vb, vi, vj] = T.float32(0)
                Y[vb, vi, vj] += A[vb, vi, vk] * B[vb, vk, vj]

        for b, i, j in T.grid(16, 128, 128):
            with T.block("C"):
                vb, vi, vj = T.axis.remap("SSS", [b, i, j])
                C[vb, vi, vj] = T.max(Y[vb, vi, vj], T.float32(0))


# Exercise 1: broadcast matrix add
def test_myadd():
    a = np.arange(16).reshape(4, 4)
    b = np.arange(4, 0, -1).reshape(4)

    # numpy version
    c_np = a + b

    # tir version
    rt_lib = tvm.build(MyAdd, target="llvm")
    a_tvm = tvm.nd.array(a)
    b_tvm = tvm.nd.array(b)
    c_tvm = tvm.nd.array(np.empty((4, 4), dtype=np.int64))
    rt_lib["add"](a_tvm, b_tvm, c_tvm)
    # compare result
    np.testing.assert_allclose(c_tvm.numpy(), c_np, rtol=1e-5)


# Exercise 2: convolution
def test_myconv():
    N, CI, H, W, CO, K = 1, 1, 8, 8, 2, 3
    OUT_H, OUT_W = H - K + 1, W - K + 1
    data = np.arange(N * CI * H * W).reshape(N, CI, H, W)
    weight = np.arange(CO * CI * K * K).reshape(CO, CI, K, K)

    # torch version
    import torch

    data_torch = torch.Tensor(data)
    weight_torch = torch.Tensor(weight)
    conv_torch = torch.nn.functional.conv2d(data_torch, weight_torch)
    conv_torch = conv_torch.numpy().astype(np.int64)

    rt_lib = tvm.build(MyConv, target="llvm")
    data_tvm = tvm.nd.array(data)
    weight_tvm = tvm.nd.array(weight)
    conv_tvm = tvm.nd.array(np.empty((N, CO, OUT_H, OUT_W), dtype=np.int64))
    rt_lib["conv"](data_tvm, weight_tvm, conv_tvm)
    np.testing.assert_allclose(conv_tvm.numpy(), conv_torch, rtol=1e-5)


# Exercise 3: batch matrix multiply
def test_mybnn():
    sch = tvm.tir.Schedule(MyBmmRelu)
    print(sch.mod.script())
    print("*****************")

    # Step 1. Get blocks
    Y = sch.get_block("Y", func_name="bmm_relu")

    # Step 2. Get loops
    b, i, j, k = sch.get_loops(Y)

    # Step 3. Organize the loops
    j0, j1 = sch.split(j, factors=[None, 8])

    sch.reorder(b, i, j0, k, j1)
    sch.parallel(b)

    C = sch.get_block("C", "bmm_relu")

    # move block C into inner loop j0
    sch.reverse_compute_at(C, j0)

    # Step 4. decompose reduction

    # split init and update
    sch.decompose_reduction(Y, k)
    # C = sch.get_block("C", "bmm_relu")
    Y_init = sch.get_block("Y_init", "bmm_relu")
    ax0_init = sch.get_loops(Y_init)
    sch.vectorize(ax0_init[3])

    ax0 = sch.get_loops(C)
    sch.vectorize(ax0[3])

    k0, k1 = sch.split(k, factors=[None, 4])
    sch.unroll(k1)

    print(sch.mod.script())
    print("****************")

    # evaluate performance
    before_rt_lib = tvm.build(MyBmmRelu, target="llvm")
    after_rt_lib = tvm.build(sch.mod, target="llvm")
    a_tvm = tvm.nd.array(np.random.rand(16, 128, 128).astype("float32"))
    b_tvm = tvm.nd.array(np.random.rand(16, 128, 128).astype("float32"))
    c_tvm = tvm.nd.array(np.random.rand(16, 128, 128).astype("float32"))
    after_rt_lib["bmm_relu"](a_tvm, b_tvm, c_tvm)
    before_timer = before_rt_lib.time_evaluator("bmm_relu", tvm.cpu())
    print("Before transformation:")
    print(before_timer(a_tvm, b_tvm, c_tvm))

    f_timer = after_rt_lib.time_evaluator("bmm_relu", tvm.cpu())
    print("After transformation:")
    print(f_timer(a_tvm, b_tvm, c_tvm))


if __name__ == "__main__":
    test_myadd()
    # Now we report munmap_chunk error here, and dont konw how to solve it.
    # test_myconv()
    test_mybnn()
