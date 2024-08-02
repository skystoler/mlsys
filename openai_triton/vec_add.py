import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr,
               y_ptr,
               output_ptr,
               n_elements, # Size of vector
               BLOCK_SIZE: tl.constexpr, # Each program should handle the number of elements
               ): 
    pid = tl.program_id(axis=0) # We use one dimension here, so axis = 0
    block_start = pid * BLOCK_SIZE
    offset = block_start + tl.arange(0, BLOCK_SIZE) # Offset is a list of pointers
    mask = offset < n_elements # Mask to avoid access out of range
    x = tl.load(x_ptr + offset, mask=mask)
    y = tl.load(y_ptr + offset, mask=mask)
    output = x + y
    tl.store(output_ptr + offset, output, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = output.numel() # Tensor.numel(): multiply every dimension
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE = 1024)
    return output

# Bechmark for testing performance of torch and triton
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # 用作绘图x轴的参数名。
        x_vals=[2**i for i in range(12, 28, 1)],  # `x_name`的不同可能值。
        x_log=True,  # x轴是对数的。
        line_arg='provider',  # 其值对应于图中不同线条的参数名。
        line_vals=['triton', 'torch'],  # `line_arg`的可能值。
        line_names=['Triton', 'Torch'],  # 线条的标签名称。
        styles=[('blue', '-'), ('green', '-')],  # 线条样式。
        ylabel='GB/s',  # y轴的标签名称。
        plot_name='vector-add-performance',  # 绘图的名称。也用作保存绘图的文件名。
        args={},  # 不在`x_names`和`y_name`中的函数参数的值。
    ))

def benchmark(size, provider):
    x = torch.rand(size, device='cuda', dtype=torch.float32)
    y = torch.rand(size, device='cuda', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y), quantiles=quantiles)
    gbps = lambda ms: 12 * size / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)

if __name__ == '__main__':
    size = 38240
    x = torch.rand(size, device='cuda')
    y = torch.rand(size, device='cuda')
    output_torch = x + y
    output_triton = add(x, y)
    assert torch.allclose(output_torch, output_triton)

    # (To do) Should learn how it works: https://github.com/triton-lang/triton/blob/main/python/triton/testing.py#L84
    benchmark.run(print_data=True, show_plots=True)