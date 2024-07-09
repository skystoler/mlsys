import numpy as np
import tvm
from tvm import tir
from tvn.script import tir as T


@tvm.script.ir_module
class MatmulModule:
    @T.prim_func
    def main(
        X: T.Buffer[(1024, 1024), "float16"],
        Y: T.Buffer[(1024, 1024), "float16"],
        Z: T.Buffer[(1024, 1024), "float32"],
    ) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i, j, k in T.grid(1024, 1024, 1024):
            with T.block("matmul"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    Z[vi, vj] = T.float32(0)
                # We cast data from fp16 to fp32 because tensore core load fp16 but calculate fp32
                Z[vi, vj] += T.cast(X[vi, vk], "float32") * T.cast(Y[vj, vk], "float32")


def register_intrinsic():
    @T.prim_func
    def wmma_load_a_desc(a: T.handle, c: T.handle) -> None:
        A = T.match_buffer(
            a, (16, 16), "float16", align=128, offset_factor=16, scope="shared"
        )
        C = T.match_buffer(
            c, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.matrix_a"
        )

        with T.block("root"):
            T.reads(A[0:16, 0:16])
            T.writes(C[0:16, 0:16])
            for i, j in T.grid(16, 16):
                with T.block("load"):
                    vii, vjj = T.axis.remap("SS", [i, j])
                    C[vii, vjj] = A[vii, vjj]

    @T.prim_func
    def wmma_load_a_impl(a: T.handle, c: T.handle) -> None:
        s1 = T.var("int32")
        s0 = T.var("int32")
        A = T.match_buffer(
            a,
            (16, 16),
            "float16",
            align=128,
            offset_factor=16,
            scope="shared",
            strides=[s1, s0],
        )
        C = T.match_buffer(
            c, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.matrix_a"
        )

        with T.block("root"):
            T.reads(A[0:16, 0:16])
            T.writes(C[0:16, 0:16])
            T.evaluate(
                T.tvm_load_matrix_sync(
                    C.data,
                    16,
                    16,
                    16,
                    C.elem_offset // 256
                    + T.floordiv(T.floormod(C.elem_offset, 256), 16),
                    A.access_ptr("r"),
                    s1,
                    "row_major",
                    dtype="handle",
                )
            )

    @T.prim_func
    def wmma_load_b_desc(a: T.handle, c: T.handle) -> None:
        A = T.match_buffer(
            a, (16, 16), "float16", align=128, offset_factor=16, scope="shared"
        )
        C = T.match_buffer(
            c, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.matrix_b"
        )

        with T.block("root"):
            T.reads(A[0:16, 0:16])
            T.writes(C[0:16, 0:16])
            for i, j in T.grid(16, 16):
                with T.block("load"):
                    vii, vjj = T.axis.remap("SS", [i, j])
                    C[vii, vjj] = A[vii, vjj]

    @T.prim_func
    def wmma_load_b_impl(a: T.handle, c: T.handle) -> None:
        s1 = T.var("int32")
        s0 = T.var("int32")
        A = T.match_buffer(
            a,
            (16, 16),
            "float16",
            align=128,
            offset_factor=16,
            scope="shared",
            strides=[s1, s0],
        )
        C = T.match_buffer(
            c, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.matrix_b"
        )

        with T.block("root"):
            T.reads(A[0:16, 0:16])
            T.writes(C[0:16, 0:16])
            T.evaluate(
                T.tvm_load_matrix_sync(
                    C.data,
                    16,
                    16,
                    16,
                    C.elem_offset // 256
                    + T.floordiv(T.floormod(C.elem_offset, 256), 16),
                    A.access_ptr("r"),
                    s1,
                    "col_major",
                    dtype="handle",
                )
            )

    @T.prim_func
    def wmma_sync_desc(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(
            a, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.matrix_a"
        )
        B = T.match_buffer(
            b, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.matrix_b"
        )
        C = T.match_buffer(
            c,
            (16, 16),
            "float32",
            align=128,
            offset_factor=16,
            scope="wmma.accumulator",
        )

        with T.block("root"):
            T.reads(C[0:16, 0:16], A[0:16, 0:16], B[0:16, 0:16])
            T.writes(C[0:16, 0:16])
            for i, j, k in T.grid(16, 16, 16):
                with T.block(""):
                    vii, vjj, vkk = T.axis.remap("SSR", [i, j, k])
                    C[vii, vjj] += T.cast(A[vii, vkk], "float32") * T.cast(
                        B[vjj, vkk], "float32"
                    )

    @T.prim_func
    def wmma_sync_impl(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(
            a, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.matrix_a"
        )
        B = T.match_buffer(
            b, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.matrix_b"
        )
        C = T.match_buffer(
            c,
            (16, 16),
            "float32",
            align=128,
            offset_factor=16,
            scope="wmma.accumulator",
        )

        with T.block("root"):
            T.reads(C[0:16, 0:16], A[0:16, 0:16], B[0:16, 0:16])
            T.writes(C[0:16, 0:16])
            T.evaluate(
                T.tvm_mma_sync(
                    C.data,
                    C.elem_offset // 256
                    + T.floordiv(T.floormod(C.elem_offset, 256), 16),
                    A.data,
                    A.elem_offset // 256
                    + T.floordiv(T.floormod(A.elem_offset, 256), 16),
                    B.data,
                    B.elem_offset // 256
                    + T.floordiv(T.floormod(B.elem_offset, 256), 16),
                    C.data,
                    C.elem_offset // 256
                    + T.floordiv(T.floormod(C.elem_offset, 256), 16),
                    dtype="handle",
                )
            )

    @T.prim_func
    def wmma_fill_desc(c: T.handle) -> None:
        C = T.match_buffer(
            c,
            (16, 16),
            "float32",
            align=128,
            offset_factor=16,
            scope="wmma.accumulator",
        )

        with T.block("root"):
            T.reads()
            T.writes(C[0:16, 0:16])
            for i, j in T.grid(16, 16):
                with T.block("init"):
                    vii, vjj = T.axis.remap("SS", [i, j])
                    C[vii, vjj] = T.float32(0)

    @T.prim_func
    def wmma_fill_impl(c: T.handle) -> None:
        C = T.match_buffer(
            c,
            (16, 16),
            "float32",
            align=128,
            offset_factor=16,
            scope="wmma.accumulator",
        )
        with T.block("root"):
            T.reads()
            T.writes(C[0:16, 0:16])
            T.evaluate(
                T.tvm_fill_fragment(
                    C.data,
                    16,
                    16,
                    16,
                    C.elem_offset // 256
                    + T.floordiv(T.floormod(C.elem_offset, 256), 16),
                    T.float32(0),
                    dtype="handle",
                )
            )

    @T.prim_func
    def wmma_store_desc(a: T.handle, c: T.handle) -> None:
        A = T.match_buffer(
            a,
            (16, 16),
            "float32",
            align=128,
            offset_factor=16,
            scope="wmma.accumulator",
        )
        C = T.match_buffer(
            c, (16, 16), "float32", align=128, offset_factor=16, scope="global"
        )
        with T.block("root"):
            T.reads(A[0:16, 0:16])
            T.writes(C[0:16, 0:16])
            for i, j in T.grid(16, 16):
                with T.block("store"):
                    vii, vjj = T.axis.remap("SS", [i, j])
                    C[vii, vjj] = A[vii, vjj]

    @T.prim_func
    def wmma_store_impl(a: T.handle, c: T.handle) -> None:
        s1 = T.var("int32")
        s0 = T.var("int32")
        A = T.match_buffer(
            a,
            (16, 16),
            "float32",
            align=128,
            offset_factor=16,
            scope="wmma.accumulator",
        )
        C = T.match_buffer(
            c,
            (16, 16),
            "float32",
            align=128,
            offset_factor=16,
            scope="global",
            strides=[s1, s0],
        )
        with T.block("root"):
            T.reads(A[0:16, 0:16])
            T.writes(C[0:16, 0:16])
            T.evaluate(
                T.tvm_store_matrix_sync(
                    A.data,
                    16,
                    16,
                    16,
                    A.elem_offset // 256
                    + T.floordiv(T.floormod(A.elem_offset, 256), 16),
                    C.access_ptr("w"),
                    s1,
                    "row_major",
                    dtype="handle",
                )
            )

    try:
        # handle exception if we register multi times
        tir.TensorIntrin.register("wmma_load_a", wmma_load_a_desc, wmma_load_a_impl)
        tir.TensorIntrin.register("wmma_load_b", wmma_load_b_desc, wmma_load_b_impl)
        tir.TensorIntrin.register("wmma_sync", wmma_sync_desc, wmma_sync_impl)
        tir.TensorIntrin.register("wmma_fill", wmma_fill_desc, wmma_fill_impl)
        tir.TensorIntrin.register("wmma_store", wmma_store_desc, wmma_store_impl)
    except ValueError:
        pass


# Split and bind loops to threads
def test_tensor_core():
    sch = tir.Schedule(MatmulModule)
    block = sch.get_block("matmul")
    i, j, k = sch.get_loops(block)

    # We split factors to 16 because wmma is suitable for 16*16*16
    i, ii = sch.split(i, factors=[None, 16])
    j, ji = sch.split(j, factors=[None, 16])
    k, ki = sch.split(k, factors=[None, 16])
    sch.reorder(i, j, k, ii, ji, ki)
    wmma_sync = sch.blockize(loop=ii)
    sch.mod.show()

    i0, i1, i2 = sch.split(i, factors=[8, 4, 2])
    j0, j1, j2 = sch.split(j, factors=[8, 4, 2])
    k0, k1, k2 = sch.split(k, factors=[16, 2, 2])

    sch.reorder(i0, j0, i1, j1, k0, k1, i2, j2, k2)
    bx = sch.fuse(i0, j0)
    sch.bind(bx, "blockIdx.x")
    ty = sch.fuse(i1, j1)
    sch.bind(ty, "threadIdx.y")
    # We can't bind to `threadIdx.x` since we have warp-level operators under the loop
    sch.mod.show()

    # We cache A and B to shared memory
    X_shared = sch.cache_read(wmma_sync, read_buffer_index=0, storage_scope="shared")
    Y_shared = sch.cache_read(wmma_sync, read_buffer_index=1, storage_scope="shared")

    def schedule_shared(block):
        sch.compute_at(block, k0)
        x, y = sch.get_loops(block)[-2:]
        fused = sch.fuse(x, y)
        x0, x1, x2, x3 = sch.split(fused, factors=[None, 16, 32, 8])
        sch.bind(x1, "threadIdx.y")
        # here we must bind threadIdx.x == 32 to satisfy the requirements of warp-level operation.
        sch.bind(x2, "threadIdx.x")
        sch.vectorize(x3)

    schedule_shared(X_shared)
    schedule_shared(Y_shared)
    sch.mod.show()

    # cache input data
    X_local = sch.cache_read(wmma_sync, 0, storage_scope="wmma.matrix_a")
    Y_local = sch.cache_read(wmma_sync, 1, storage_scope="wmma.matrix_b")
    sch.compute_at(X_local, k1)
    sch.compute_at(Y_local, k1)

    # cache output data
    write_back_block = sch.cache_write(wmma_sync, 0, storage_scope="wmma.accumulator")
    sch.reverse_compute_at(write_back_block, ty)
    sch.mod.show()

    def schedule_copy(block):
        x, y = sch.get_loops(block)[-2:]
        x0, x1 = sch.split(x, factors=[None, 16])
        y0, y1 = sch.split(y, factors=[None, 16])
        sch.reorder(x0, y0, x1, y1)

    schedule_copy(X_local)
    schedule_copy(Y_local)
    schedule_copy(write_back_block)
    sch.mod.show()

    # Tensorize
    init = sch.decompose_reduction(wmma_sync, k0)
    sch.tensorize(sch.get_loops(X_local)[-2], "wmma_load_a")
    sch.tensorize(sch.get_loops(Y_local)[-2], "wmma_load_b")
    sch.tensorize(init, "wmma_fill")
    sch.tensorize(wmma_sync, "wmma_sync")
    sch.tensorize(sch.get_loops(write_back_block)[-2], "wmma_store")
    sch.mod.show()

    # Evaluate
    rt_mod = tvm.build(sch.mod, target="cuda")

    dev = tvm.cuda()
    num_flop = 1024**3 * 2
    A_np = np.random.randn(1024, 1024).astype("float16")
    B_np = np.random.randn(1024, 1024).astype("float16")
    C_np = A_np.astype("float32") @ (B_np.astype("float32").T)

    A_nd = tvm.nd.array(A_np, dev)
    B_nd = tvm.nd.array(B_np, dev)
    C_nd = tvm.nd.array(np.empty((1024, 1024), dtype="float32"), dev)

    rt_mod(A_nd, B_nd, C_nd)
    np.testing.assert_allclose(C_np, C_nd.numpy(), rtol=1e-3, atol=1e-3)

    evaluator = rt_mod.time_evaluator("main", dev, number=10)
    print(
        "Performance: %f GFLOPS" % (num_flop / evaluator(A_nd, B_nd, C_nd).mean / 1e9)
    )


if __name__ == "__main__":
    register_intrinsic()
    test_tensor_core()
