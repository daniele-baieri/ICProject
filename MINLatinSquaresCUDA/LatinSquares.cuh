#pragma once

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>


__global__ void checkLatinSquare(bool* matrices, int* topology, bool* conf);

__global__ void setupRandState(curandState* state);