import triton
import torch
import triton.language as tl

"""
# Realize the algorithm of BLOCK-level matmul below.
# Do in parallel
for m in range(0, M, BLOCK_SIZE_M):
  # Do in parallel
  for n in range(0, N, BLOCK_SIZE_N):
    acc = zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=float32)
    for k in range(0, K, BLOCK_SIZE_K):
      a = A[m : m+BLOCK_SIZE_M, k : k+BLOCK_SIZE_K]
      b = B[k : k+BLOCK_SIZE_K, n : n+BLOCK_SIZE_N]
      acc += dot(a, b)
    C[m : m+BLOCK_SIZE_M, n : n+BLOCK_SIZE_N] = acc
"""

# Func decorated by 'triton.jit' uses 'triton.autotune' to change configuration.
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
    ],
    key=['M', 'N', 'K'],
)

@triton.jit
def matmul_kernel(A_ptr,
                  B_ptr,
                  C_ptr,
                  A_stride_m, A_stride_k,
                  B_stride_k, B_stride_n,
                  C_stride_m, C_stride_n,
                  M, N, K,
                  BLOCK_SIZE_M: tl.constexpr,
                  BLOCK_SIZE_K: tl.constexpr,
                  BLOCK_SIZE_N: tl.constexpr,
                  GROUP_SIZE_M: tl.constexpr,
                  ACTIVATION: tl.constexpr 
                ):
    # Each program calculate a sub matrix [BLOCK_SIZE_M, BLOCK_SIZE_N]
    """
    Implementation without using grouped ordeing

    pid = tl.program_id(axis = 0)
    grid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M;
    grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N;
    pid_m = pid / grid_n;
    pid_n = pid % grid_n; 
    """
    # Implementation using grouped ordeing to reuse L2 cache
    pid = tl.program_id(axis=0)
    
    # L2 cache optimization implemented by group ordering
    # Perfomance can be improved by 10 percent
    # See https://mp.weixin.qq.com/s/RMR_n1n6nBqpdMl6tdd7pQ for better understanding

    # cdiv means ceiling division to get M / BLOCK_SIZE_M
    # get how many loops on direction M and N
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    # the number of programs in the group
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    # id of the group which the program is in
    group_id = pid // num_pid_in_group
    # id of the first program in the group
    first_pid_m = group_id * GROUP_SIZE_M
    # If num_pid_m is not divisible, the last group is smaller.
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    # row id in grid
    pid_m = first_pid_m + (pid % group_size_m)
    # col id in grid
    pid_n = (pid % num_pid_in_group) // group_size_m

    offset_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offset_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offset_k = tl.arange(0, BLOCK_SIZE_K)

    # a_ptrs points to [BLOCK_SIZE_M, BLOCK_SIZE_K]
    # b_ptrs points to [BLOCK_SIZE_K, BLOCK_SIZE_N]
    a_ptrs = A_ptr + (offset_am[:, None] * A_stride_m + offset_k[None, :] * A_stride_k)
    b_ptrs = B_ptr + (offset_bn[:, None] * B_stride_k + offset_k[None, :] * B_stride_n)

    # Use float32 to calculate to get higher precision
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype = tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask = offset_k[None, :] < K - k * BLOCK_SIZE_K, other= 0.0)
        b = tl.load(b_ptrs, mask = offset_k[:, None] < K - k * BLOCK_SIZE_K, other= 0.0)
        accumulator += tl.dot(a, b)
        
        a_ptrs += BLOCK_SIZE_K * A_stride_k;
        b_ptrs += BLOCK_SIZE_K * B_stride_k;

    # Fused op
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    c = accumulator.to(tl.float16)

    # Store using mask
    offset_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + C_stride_m * offset_cm[:, None] + C_stride_n * offset_cn[None, :]
    c_mask = (offset_cm[:, None] < M) & (offset_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

# Fused op
@triton.jit
def leaky_relu(x):
    x = x + 1
    return tl.where(x > 0, 0.01 * x)

def matmul(a, b, activation=""):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device = a.device, dtype = a.dtype)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_kernel[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        ACTIVATION=activation  #
    )
    return c

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M', 'N', 'K'],  # 用作图表x轴的参数名
        x_vals=[128 * i for i in range(2, 33)],  # `x_name`的不同可能值
        line_arg='provider',  # 其值对应于图表中不同线条的参数名
        # `line_arg`的可能值
        line_vals=['cublas', 'triton'],
        # 线条的标签名称
        line_names=["cuBLAS", "Triton"],
        # 线条样式
        styles=[('green', '-'), ('blue', '-')],
        ylabel="TFLOPS",  # y轴的标签名称
        plot_name="matmul-performance",  # 图表的名称，也用作保存图表的文件名。
        args={},  # 其他参数
    ))
def benchmark(M, N, K, provider):
    # 初始化张量
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    quantiles = [0.5, 0.2, 0.8]  # 分位数
    # 如果提供者是cublas
    if provider == 'cublas':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
    # 如果提供者是triton
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), quantiles=quantiles)
    # 性能计算函数
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)

if __name__ == '__main__':
    a = torch.randn((512, 512), device='cuda', dtype=torch.float16)
    b = torch.randn((512, 512), device='cuda', dtype=torch.float16)
    triton_output = matmul(a, b)
    torch_output = torch.matmul(a, b)
    # absolute tolerance (atol): absolute(a - b) <= atol
    # relative tolerance (rtol): absolute(a - b) <= rtol * absolute(b)
    assert torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0)

    benchmark.run(show_plots=True, print_data=True)