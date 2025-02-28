import triton
import triton.language as tl
import torch

@triton.jit
def matmul_kernel(
    x_ptr, 
    y_ptr, 
    output_ptr, 
    M, 
    N, 
    K, 
    BLOCK_SIZE_M: tl.const_expr, 
    BLOCK_SIZE_N: tl.const_expr, 
    BLOCK_SIZE_K: tl.const_expr,
):
    pid_x = tl.program_id(axis = 0)
    pid_y = tl.program_id(axis = 1)
    offset_x = pid_x * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:, None]
    offset_y = pid_y * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None, :]
    z = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype = x.dtype)
    for k in range(0, K, BLOCK_SIZE_K):
        offset_x_k = tl.arange(0, BLOCK_SIZE_K)[None, :] + k
        x = tl.load(x_ptr + offset_x * K + offset_x_k, mask=(offset_x < M) & (offset_x_k < K), other = 0.0)
        
        offset_y_k = tl.arange(0, BLOCK_SIZE_K)[:, None] + k
        y = tl.load(y_ptr + offset_y_k * N + offset_y, mask=(offset_y < N) & (offset_y_k < K), other = 0.0)
        
        z = tl.dot(x, y, acc=z)
    
    tl.store(output_ptr + offset_x * N + offset_y, z, mask=(offset_x < M) & (offset_y < N))

def triton_matmul(x, y):
    M, K = x.shape
    N = y.shape[-1]
    output = torch.empty((M, N), device = x.device, dtype = x.dtype)
    BLOCK_SIZE_M = 8
    BLOCK_SIZE_N = 8
    BLOCK_SIZE_K = 4
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N), 1)
    matmul_kernel[grid](x, y, output, M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)
    return output
    
if __name__ == '__main__':
    x = torch.rand((16, 32), device='cuda')
    y = torch.rand((32, 16), device='cuda')
    output = torch.matmul(x, y)
    output_triton = triton_matmul(x, y)
    assert torch.allclose(output, output_triton)