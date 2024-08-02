import triton
import torch
import triton.language as tl

@torch.jit.script
def native_softmax(x):
    # Suppose x is a [M, N] tensor
    """
    if x = [[1,2,3,4]
            [4,3,2,1]
            [2,5,1,6]],
    x.max(dim = 1) = [4, 4, 6],
    x.max(dim = 1)[0] = [4, 4, 6],
    x_max[:, None] = [[4], [4], [6]],
    z = [[-3, -2, -1, 0]
         [0, -1, -2, -3]
         [-4, -1, -5, 0]],
    denominator = [sum_row1, sum_row2, sum_row3]
    denominator[:, None] = [[sum_row1], [sum_row2], [sum_row3]]
    """
    # read MN, write M
    x_max = x.max(dim = 1)[0]
    # read MN + M, write MN
    z = x - x_max[:, None] # Avoid overflow
    # read MN, write MN
    numerator = torch.exp(z)
    # read MN, write M 
    denominator = numerator.sum(dim = 1)
    # read MN + M, write MN
    ret = numerator / denominator[:, None]
    # sum 5MN + 2M; wrirte 3MN + 2M
    # aim to wite 2MN and write 2MN, for 4x speed up
    return ret

@triton.jit
def softmax_kernel( input_ptr,
                    output_ptr,
                    input_row_stride,
                    output_row_stride,
                    n_cols,
                    BLOCK_SIZE:tl.constexpr,
                    ):
    # read / load
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offset = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offset
    row = tl.load(input_ptrs, mask = col_offset < n_cols, other = -float('inf'))

    # calculate
    row_minus_max = row - tl.max(row, axis = 0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis = 0)
    softmax_output = numerator / denominator

    # write / store
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr+ col_offset
    tl.store(output_ptrs, softmax_output, mask = col_offset < n_cols)

def softmax(x):
    n_rows, n_cols = x.shape
    # BLOCK_SIZE is set to mininal 2's pow which is more than 2
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # A trick to use more threads in every row
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16

    y = torch.empty_like(x)

    softmax_kernel(x, 
                   y, 
                   x.stride(0), 
                   y.stride(0), 
                   n_cols, 
                   num_warps=num_warps,
                   BLOCK_SIZE=BLOCK_SIZE
                   )
    return y

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # 用作绘图x轴的参数名
        x_vals=[128 * i for i in range(2, 100)],  # `x_name`的不同可能值
        line_arg='provider',  # 其值对应于图中不同线条的参数名
        line_vals=[
            'triton',
            'torch-native',
            'torch-jit',
        ],  # `line_arg`的可能值
        line_names=[
            "Triton",
            "Torch (原生)",
            "Torch (jit)",
        ],  # 线条的标签名称
        styles=[('blue', '-'), ('green', '-'), ('green', '--')],  # 线条样式
        ylabel="GB/s",  # y轴的标签名称
        plot_name="softmax-performance",  # 绘图的名称。也用作保存绘图的文件名。
        args={'M': 4096},  # 不在`x_names`和`y_name`中的函数参数的值
    ))
def benchmark(M, N, provider):
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch-native':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: softmax(x), quantiles=quantiles)
    if provider == 'torch-jit':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: naive_softmax(x), quantiles=quantiles)
    gbps = lambda ms: 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)

if __name__ == '__main__':
    x = torch.randn(1823,781, device = 'cuda')
    y_torch = torch.softmax(x, axis = 1)
    y_triton = softmax(x)
    assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)

    benchmark.run(show_plots=True, print_data=True)