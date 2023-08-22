import numpy as np
import tvm
from tvm.script import tir as T


@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def main(
        A: T.Buffer((128, 128), "float32"),
        B: T.Buffer((128, 128), "float32"),
        C: T.Buffer((128, 128), "float32"),
    ):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i, j, k in T.grid(128, 128, 128):
            with T.block("C"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


# Here we add some random elements to generate our transform
def stochastic_schedule_mm(sch: tvm.tir.Schedule):
    block_C = sch.get_block("C", "main")
    i, j, k = sch.get_loops(block=block_C)

    # use random number as the value of j_factor
    j_factors = sch.sample_perfect_tile(loop=j, n=2)
    print(type(j_factors[0]))

    j_0, j_1 = sch.split(loop=j, factors=j_factors)
    sch.reorder(i, j_0, k, j_1)
    print(sch.trace)

    sch.decompose_reduction(block_C, k)
    return sch


def random_search(mod: tvm.IRModule, num_trials=5):
    dtype = "float32"
    a_np = np.random.rand(128, 128).astype(dtype)
    b_np = np.random.rand(128, 128).astype(dtype)
    a_nd = tvm.nd.array(a_np)
    b_nd = tvm.nd.array(b_np)
    c_nd = tvm.nd.empty((128, 128), dtype="float32")

    best_result = None
    best_sch = None

    for i in range(num_trials):
        sch = stochastic_schedule_mm(tvm.tir.Schedule(mod))
        lib = tvm.build(sch.mod, target="llvm")
        f_timer_after = lib.time_evaluator("main", tvm.cpu())
        result = f_timer_after(a_nd, b_nd, c_nd).mean

        print("=====Attempt %d, time-cost: %.3f ms====" % (i, result * 1000))
        print(sch.trace)

        # book keep the best result so far
        if best_result is None or result < best_result:
            best_result = result
            best_sch = sch

    return best_sch


# We use evolutionary search and cost model here
from tvm import meta_schedule as ms


def meta_schedule():
    dtype = "float32"
    a_np = np.random.rand(128, 128).astype(dtype)
    b_np = np.random.rand(128, 128).astype(dtype)
    a_nd = tvm.nd.array(a_np)
    b_nd = tvm.nd.array(b_np)
    c_nd = tvm.nd.empty((128, 128), dtype="float32")

    database = ms.tune_tir(
        mod=MyModule,
        target="llvm --num-cores=1",
        max_trials_global=64,
        num_trials_per_iter=64,
        space=ms.space_generator.ScheduleFn(stochastic_schedule_mm),
        work_dir="./tune_tmp",
        task_name="main",
    )

    sch = ms.tir_integration.compile_tir(database, MyModule, "llvm --num-cores=1")
    print(sch.mod.script())
    lib = tvm.build(sch.mod, target="llvm")
    f_timer_after = lib.time_evaluator("main", tvm.cpu())
    print(
        "Time cost of MyModule after tuning: %.3f ms"
        % (f_timer_after(a_nd, b_nd, c_nd).mean * 1000)
    )


if __name__ == "__main__":
    sch = random_search(MyModule)
    meta_schedule()
