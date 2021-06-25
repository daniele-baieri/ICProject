﻿#pragma once

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>


__global__ void check_latin_square(bool* matrices, int* topology, bool* conf, bool* is_latin_square, int* perm);

__global__ void setup_rand_state(curandState* state);