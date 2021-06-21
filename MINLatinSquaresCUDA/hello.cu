#include <stdio.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <helper_functions.h>
#include <helper_cuda.h>

__global__ void helloCUDA(float f) {
    printf("hello thread %d, arg = %f", threadIdx.x, f);
}

int main() {
    helloCUDA << <1, 5 >> > (0.4f);
    cudaDeviceSynchronize();
    return 0;
}