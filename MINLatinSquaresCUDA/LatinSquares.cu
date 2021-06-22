#include "LatinSquares.cuh"
#include <math.h>


__global__ void setupRandState(curandState* state) {

	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	curand_init(1234, idx, 0, &state[idx]);

}

__global__ void checkLatinSquare(bool* matrices, int* topology, bool* conf) {

	curandState* d_state;
	int idx = threadIdx.x + blockDim.x * blockIdx.x;  // fix when calling
	curand_init(1234, idx, 0, &d_state[idx]);

	for (int i = 0; i < 8; i++) {
		for (int j = 0; j < 7; j++) {
			conf[i * 7 + j] = (int)truncf(curand_uniform(d_state));
		}
	}

	int perm[16 * 16];
	bool xor[8 * 7];
	int curr_perm[16 * 7];
	for (int m = 0; m < 16; m++) {
		bool* to_xor = &matrices[m * 8 * 7];
		for (int i = 0; i < 8; i++) {
			for (int j = 0; j < 7; j++) {
				auto idx = i * 7 + j;
				xor [idx] = (!conf[idx] != !to_xor[idx]);
			}
		}

		for (int stage = 0; stage < 7; stage++) {
			for (int sw = 0; sw < 8; sw++) {
				// apply switches
				int up_port = sw * 2;
				int low_port = sw * 2 + 1;
				if (stage == 0) {
					curr_perm[up_port * 7 + stage] = up_port;
					curr_perm[low_port * 7 + stage] = low_port;
				}

				bool setting = xor [sw * 7 + stage];
				if (setting) {
					int lp = curr_perm[low_port * 7 + stage];
					curr_perm[low_port * 7 + stage] = curr_perm[up_port * 7 + stage];
					curr_perm[up_port * 7 + stage] = lp;
				}
			}
			for (int port = 0; port < 16 && stage < 6; port++) {
				// apply topology
				curr_perm[topology[port * 6 + stage] * 7 + (stage + 1)] = curr_perm[port * 7 + stage];
			}
		}
	}

}