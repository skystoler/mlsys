#include "cuda.h"

/*
gemm: A(842 * 64)* B(64 * 128) = C(842 * 128)
 */
void optimize_gemm(float* input, float* output, const int M, const int N, const int K){
    cuda_init();
    int tile_size = 32;
    optimize_gemm_kernel<<<dim3(M, N), dim3(tile_size, tile_size)>>>(input, output, M, N, K);
    cuda_free();
}

__global__ void optimize_gemm_kernel(float* input, float* output, const int M, const int N, const int K){
    int i = blockIdx.x *  + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i < M && j < N){
        float sum = 0;
        __shared__ float cache[32] = {0};
        for(int k = 0; k < K; k ++){
            cache[] = input[i * K + k] * input[j * K + k];
        }
        output[i * N + j] = sum;
    }
}

int main(){
    int M = 842;
    int N = 128;
    int K = 64;
    float *input = (float*)malloc(M * K * sizeof(float));
    float *output = (float*)malloc(M * N * sizeof(float));
    optimize_gemm(input, output, M, N, K);
}