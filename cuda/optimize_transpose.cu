#include <stdio.h>
#include <iostream>
#include <cuda.h>

void optimize_transpose(float *input, float *output, int m, int n) {
    int block_size = 32;
    dim3 block(block_size, block_size);
    dim3 grid(n / block_size, m / block_size);
    optimize_transpose_kernel<<<grid, block>>>(input, output, m, n);
}

__global__ void optimize_transpose_kernel(float *input, float *output, const int n) {
    int block_size = 32;
    int row = blockIdx.y * block_size + threadIdx.y;
    int col = blockIdx.x * block_size + threadIdx.x;

    // Plus 1 here to avoid bank conflict
    __shared__ float tile[block_size][block_size + 1];
    if (row < n && col < n) {
        tile[threadIdx.y][threadIdx.x] = input[row * n + col];
    }
    __syncthreads();

    // use merged access to write the result to global memory
    row = blockIdx.x * block_size + threadIdx.y;
    col = blockIdx.y * block_size + threadIdx.x;
    if (row < n && col < n) {
        output[row * n + col] = tile[threadIdx.x][threadIdx.y];
    }
}