﻿#include <stdio.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

__global__ void helloCUDA(
    unsigned int N,
    bool conf[],
    bool is_latin_square,
    bool out_conf[]
) {

}


/*
int main() {
    
    helloCUDA << <1, 5 >> > (0.4f);
    cudaDeviceSynchronize();
    return 0;
    
}
*/