# This program introduces schedule to avoid writing programs repeatly
# and help to optimize for convinience

import tvm

from chapter_1 import MyModule


def display_module(module, transform_type="Unknown", transform=True):
    if transform is True:
        print("Module Display after transform " + transform_type + ":")
    else:
        print("Original Module Display:")
    print(module.script())
    print("**************************")


if __name__ == "__main__":
    # Before transform
    display_module(MyModule, transform=False)

    # Start to schedule
    sch = tvm.tir.Schedule(MyModule)
    block_Y = sch.get_block("Y", func_name="mm_relu")
    i, j, k = sch.get_loops(block_Y)

    # None is like -1 in shape inference
    j0, j1 = sch.split(j, factors=[None, 4])

    display_module(sch.mod, transform_type="split")

    # Reorder create loops like this:
    # for i in range(128):
    #     for j0 in range(32):
    #         for k in range(128):
    #             for j1 in range(4):
    sch.reorder(j0, k, j1)

    display_module(sch.mod, transform_type="reorder")

    block_C = sch.get_block("C", "mm_relu")
    sch.reverse_compute_at(block_C, j0)

    display_module(sch.mod, transform_type="reverse_compute_at")

    # Split initial and reduction update of Y
    # This is originally implict, we just make it explicit to show the effect
    sch.decompose_reduction(block_Y, k)
    display_module(sch.mod, transform_type="decompose_reduction")
