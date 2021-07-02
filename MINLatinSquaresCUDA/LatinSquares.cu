#include "LatinSquares.cuh"
#include "Utils.h"
#include "MIN.h"
#include <math.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>

#define OUTFILE "./Results/results.txt"
#define DELIMITER "-----------"


__global__ void setup_rand_state(curandState* state) {

	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	curand_init(1234, idx, 0, &state[idx]);

}

__global__ void check_latin_square(curandState* d_state, bool* matrices, int* topology, bool* conf, bool* is_latin_square, int* perm) {

	int t_idx = threadIdx.x + blockDim.x * blockIdx.x;  

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

void cuda_handle_error() {
	auto err = cudaGetLastError();
	printf("%s: %s\n", cudaGetErrorName(err), cudaGetErrorString(err));
}

void cuda_main() {

	int* TOP = new int[16 * 6];
	make_butterfly_butterfly_topology(TOP);
	// print_int_matrix(TOP, 16, 6);

	bool* char_mat = new bool[16 * 8 * 7];
	make_characteristic_matrices(0, char_mat);

	bool* matrices;
	int* topology;
	bool* conf;
	bool* is_latin_square;
	int* perm;
	curandState* dev_states;

	const int GRID = 10;
	const int BLOCK = 10;

	cudaMalloc((void**)&matrices, (16 * 8 * 7) * sizeof(bool));
	cudaMalloc((void**)&topology, (16 * 6) * sizeof(int));
	cudaMalloc((void**)&conf, (GRID * BLOCK) * (8 * 7) * sizeof(bool));
	cudaMalloc((void**)&is_latin_square, (GRID * BLOCK) * sizeof(bool));
	cudaMalloc((void**)&perm, (GRID * BLOCK) * (16 * 16) * sizeof(int)); 
	cudaMalloc((void**)&dev_states, GRID * BLOCK * sizeof(curandState));
	cuda_handle_error();

	setup_rand_state << <GRID, BLOCK >> > (dev_states);
	cuda_handle_error();

	cudaMemcpy(matrices, char_mat, (16 * 8 * 7) * sizeof(bool), cudaMemcpyHostToDevice);
	cuda_handle_error();
	cudaMemcpy(topology, TOP, (16 * 6) * sizeof(int), cudaMemcpyHostToDevice);
	cuda_handle_error();

	delete[] TOP;
	delete[] char_mat;

	check_latin_square <<<GRID, BLOCK>>> (dev_states, matrices, topology, conf, is_latin_square, perm);
	cuda_handle_error();

	bool* output = new bool[GRID * BLOCK];
	bool* out_conf = new bool[(GRID * BLOCK) * (8 * 7)];
	int* out_perm = new int[(GRID * BLOCK) * (16 * 16)];

	cudaMemcpy(output, is_latin_square, GRID * BLOCK * sizeof(bool), cudaMemcpyDeviceToHost);
	cuda_handle_error();
	cudaMemcpy(out_conf, conf, (GRID * BLOCK) * (8 * 7) * sizeof(bool), cudaMemcpyDeviceToHost);
	cuda_handle_error();
	cudaMemcpy(out_perm, perm, (GRID * BLOCK) * (16 * 16) * sizeof(int), cudaMemcpyDeviceToHost);
	cuda_handle_error();

	FILE* fd = fopen(OUTFILE, "w+");

	int idx;
	bool out_ij;
	for (int i = 0; i < GRID; i++) {
		for (int j = 0; j < BLOCK; j++) {
			idx = i * BLOCK + j;
			out_ij = output[idx];
			fprintf(fd, "THREAD %d: OUTPUT = %d\n\n", idx, out_ij);
			for (int x = 0; x < 16; x++) {
				for (int y = 0; y < 16; y++) 
					fprintf(fd, "%2d ", out_perm[idx * 16 * 16 + x * 16 + y]);
				fprintf(fd, "| ");
				for (int y = 0; y < 7 && x < 8; y++) 
					fprintf(fd, "%d ", out_conf[idx * 8 * 7 + x * 7 + y]);
				fprintf(fd, "\n");
			}
			fprintf(fd, "\n%s\n\n", DELIMITER);
		}
	}

	fclose(fd);

	/*
	printf("Outputs for each thread:\n");
	print_bool_matrix(output, GRID, BLOCK);

	
	int r_nonzero = -1;
	int c_nonzero = -1;
	int r_zero = -1;
	int c_zero = -1;
	bool out_ij;
	for (int i = 0; i < GRID; i++) {
		for (int j = 0; j < BLOCK; j++) {
			out_ij = output[i * BLOCK + j];
			if (out_ij) {
				r_nonzero = i;
				c_nonzero = j;
			}
			else if (!out_ij) {
				r_zero = i;
				c_zero = j;
			}
			if (r_nonzero > -1 && c_nonzero > -1 && r_zero > -1 && c_zero > -1) break;
		}
	}


	printf("Configuration and permutation for a successful thread:\n");
	print_bool_matrix(&out_conf[(r_nonzero * BLOCK + c_nonzero) * 8 * 7], 8, 7);
	print_int_matrix(&out_perm[(r_nonzero * BLOCK + c_nonzero) * 16 * 16], 16, 16);

	printf("Configuration and permutation for an unsuccessful thread:\n");
	print_bool_matrix(&out_conf[(r_zero * BLOCK + c_zero) * 8 * 7], 8, 7);
	print_int_matrix(&out_perm[(r_zero * BLOCK + c_zero) * 8 * 7], 16, 16);
	*/
	
	

}
