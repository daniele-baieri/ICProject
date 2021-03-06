#pragma once

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>


__global__ void check_latin_square(curandState* state, bool* matrices, int* topology, bool* conf, bool* is_latin_square, int* perm);

__global__ void setup_rand_state(curandState* state);

__global__ void check_mols_random(curandState* state, int* perms, bool* latin_squares, bool* mols, int* pairs, bool debug);

__global__ void check_mols_complete(int* perms, bool* latin_squares, bool* mols, int* pairs, bool debug);

__global__ void check_mols_complete(int* perms_A, int* perms_B, bool* lat_sq_A, bool* lat_sq_B, bool* mols, int* pairs, bool debug);