from typing import List
import torch

def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print(">>> my_compiler() invoked:")
    print(">>> FX graph:")
    gm.graph.print_tabular()
    print(f">>> Code:\n{gm.code}")
    return gm.forward  # return a python callable

@torch.compile(backend=my_compiler)
def foo(x, y):
    return (x + y) * x

if __name__ == '__main__':
    a, b = torch.randn((3, 2), dtype=torch.fp32)
    foo(a, b)