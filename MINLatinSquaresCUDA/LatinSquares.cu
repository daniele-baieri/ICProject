#include "LatinSquares.cuh"
#include "Utils.h"
#include "MIN.h"
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>


__global__ void setup_rand_state(curandState* state) {

	int num_in_block = blockDim.x * blockDim.y;
	int tid_in_block = threadIdx.x + blockDim.x * threadIdx.y;
	int bid_in_grid = blockIdx.x + gridDim.x * blockIdx.y;
	int idx = bid_in_grid * num_in_block + tid_in_block;
	curand_init(1027, idx, 0, &state[idx]);

}

__global__ void check_latin_square(curandState* d_state, bool* matrices, int* topology, bool* conf, bool* is_latin_square, int* perm) {

	int t_idx = blockIdx.x;  

	// printf("%d\n", t_idx);

	float r;
	for (int i = 0; i < 8; i++) {
		for (int j = 0; j < 7; j++) {
			r = curand_uniform(&d_state[t_idx]);
			// printf("%f\n", round(r));
			conf[(t_idx) * 8 * 7 + i * 7 + j] = (r >= 0.5f ? true : false);
		}
	}

	bool xor [8 * 7];
	int curr_perm[16 * 7];

	for (int m = 0; m < 16; m++) {
		bool* to_xor = &matrices[m * 8 * 7];
		for (int i = 0; i < 8; i++) {
			for (int j = 0; j < 7; j++) {
				auto idx = i * 7 + j;
				xor [idx] = (!conf[t_idx * 8 * 7 + idx] != !to_xor[idx]);
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
				if (stage == 6) {
					perm[(t_idx) * 16 * 16 + m * 16 + up_port] = curr_perm[up_port * 7 + stage];
					perm[(t_idx) * 16 * 16 + m * 16 + low_port] = curr_perm[low_port * 7 + stage];
				}

			}
			for (int port = 0; port < 16 && stage < 6; port++) {
				// apply topology
				curr_perm[topology[port * 6 + stage] * 7 + (stage + 1)] = curr_perm[port * 7 + stage];
			}
		}
	}

	is_latin_square[t_idx] = true;

	int v;
	for (int i = 0; i < 16; i++) {
		for (int j = 0; j < 16; j++) {
			v = perm[(t_idx) * 16 * 16 + i * 16 + j];
			for (int r = i + 1; r < 16; r++) {
				if (perm[(t_idx) * 16 * 16 + r * 16 + j] == v) {
					is_latin_square[t_idx] = false;
					break;
				}
			}
		}
	}

}

__global__ void check_mols_random(curandState* state, int* perms, bool* latin_squares, bool* mols, int* pairs, bool debug) {

	int idx_src = blockIdx.y * gridDim.x + blockIdx.x;
	int idx_dst = idx_src * blockDim.x + threadIdx.x;
	
	int compare = (int)(curand_uniform(&state[idx_dst]) * (gridDim.x * gridDim.y));
	// printf("Thread: %d Compares %d and %d\n", idx_dst, idx_src, compare);

	pairs[idx_dst * 2] = idx_src;
	pairs[idx_dst * 2 + 1] = compare;

	if (!latin_squares[idx_src] || !latin_squares[compare]) {	// return false if one of idx_src and compare isn't a latin square
		if (debug) printf("Thread: %d -- %d or %d is not a latin square\n", idx_dst, idx_src, compare);
		mols[idx_dst] = false;
	} else if (idx_src == compare) {  // return false if random picked source latin square (can't be orthogonal to itself)
		if (debug) printf("Thread: %d -- %d = %d\n", idx_dst, idx_src, compare);
		mols[idx_dst] = false;
	} else {
		mols[idx_dst] = true;
		int* ls_A = &perms[idx_src * 16 * 16];
		int* ls_B = &perms[compare * 16 * 16];
		int a, b;
		for (int i = 0; i < 16 * 16 && mols[idx_dst]; i++) {
			a = ls_A[i];
			b = ls_B[i];
			for (int j = i + 1; j < 16 * 16; j++) {
				if (a == ls_A[j] && b == ls_B[j]) {
					if (debug) printf("Thread: %d -- LS(%d, %d) -- (%d, %d) = (%d, %d) [%d, %d]\n", idx_dst, idx_src, compare, a, b, ls_A[j], ls_B[j], i, j);
					mols[idx_dst] = false;
					break;
				}
			}
		}
	}
	
}

__global__ void check_mols_complete(int* perms, bool* latin_squares, bool* mols, int* pairs, bool debug) {

	int idx_src = blockIdx.x;
	int compare = (blockIdx.y * blockDim.x) + threadIdx.x;

	int block_id = blockIdx.y * gridDim.x + blockIdx.x;
	int idx_dst = block_id * blockDim.x + threadIdx.x;

	// printf("Thread: %d Compares %d and %d\n", idx_dst, idx_src, compare);

	pairs[idx_dst * 2] = idx_src;
	pairs[idx_dst * 2 + 1] = compare;

	if (!latin_squares[idx_src] || !latin_squares[compare]) {	// return false if one of idx_src and compare isn't a latin square
		if (debug) printf("Thread: %d -- %d or %d is not a latin square\n", idx_dst, idx_src, compare);
		mols[idx_dst] = false;
	}
	else if (idx_src <= compare) {  // can't be orthogonal to itself + break symmetry
		// if (debug) printf("Thread: %d -- %d <= %d\n", idx_dst, idx_src, compare);
		mols[idx_dst] = false;
	}
	else {
		mols[idx_dst] = true;
		int* ls_A = &perms[idx_src * 16 * 16];
		int* ls_B = &perms[compare * 16 * 16];
		int a, b;
		for (int i = 0; i < 16 * 16 && mols[idx_dst]; i++) {
			a = ls_A[i];
			b = ls_B[i];
			for (int j = i + 1; j < 16 * 16; j++) {
				if (a == ls_A[j] && b == ls_B[j]) {
					if (debug) printf(
						"Thread: %d -- LS: (A = %d, B = %d) -- [(A(%d, %d), B(%d, %d)) = (A(%d, %d), B(%d, %d)) = (%d, %d)]\n",
						idx_dst, idx_src, compare, i / 16, i % 16, i / 16, i % 16, j / 16, j % 16, j / 16, j % 16, a, b
					);
					mols[idx_dst] = false;
					break;
				}
			}
		}
	}

}


__global__ void check_mols_complete(int* perms_A, int* perms_B, bool* lat_sq_A, bool* lat_sq_B, bool* mols, int* pairs, bool debug) {

	int idx_src = blockIdx.x;
	int compare = (blockIdx.y * blockDim.x) + threadIdx.x;

	int block_id = blockIdx.y * gridDim.x + blockIdx.x;
	int idx_dst = block_id * blockDim.x + threadIdx.x;

	// printf("Thread: %d Compares %d and %d\n", idx_dst, idx_src, compare);

	pairs[idx_dst * 2] = idx_src;
	pairs[idx_dst * 2 + 1] = compare;

	if (!lat_sq_A[idx_src] || !lat_sq_B[compare]) {	// return false if one of idx_src and compare isn't a latin square
		if (debug) printf("Thread: %d -- %d (CHAR) or %d (ROT) is not a latin square\n", idx_dst, idx_src, compare);
		mols[idx_dst] = false;
	}
	else if (idx_src == compare) {  // can't be orthogonal to itself, no symmetry here
		// if (debug) printf("Thread: %d -- %d <= %d\n", idx_dst, idx_src, compare);
		mols[idx_dst] = false;
	}
	else {
		mols[idx_dst] = true;
		int* ls_A = &perms_A[idx_src * 16 * 16];
		int* ls_B = &perms_B[compare * 16 * 16];
		int a, b;
		for (int i = 0; i < 16 * 16 && mols[idx_dst]; i++) {
			a = ls_A[i];
			b = ls_B[i];
			for (int j = i + 1; j < 16 * 16; j++) {
				if (a == ls_A[j] && b == ls_B[j]) {
					if (debug) printf(
						"Thread: %d -- LS: (CHAR = %d, ROT = %d) -- [(A(%d, %d), B(%d, %d)) = (A(%d, %d), B(%d, %d)) = (%d, %d)]\n",
						idx_dst, idx_src, compare, i / 16, i % 16, i / 16, i % 16, j / 16, j % 16, j / 16, j % 16, a, b
					);
					mols[idx_dst] = false;
					break;
				}
			}
		}
	}

}