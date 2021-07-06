#include "LatinSquares.cuh"
#include "Utils.h"
#include "MIN.h"
#include <math.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <time.h>



#define OUTFILE1 "./Results/results-ls.txt"
#define OUTFILE2 "./Results/results-mols.txt"

#define CUDA_ERROR_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "%s: %s %s %d\n", cudaGetErrorName(code), cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

void cuda_handle_error() {
	auto err = cudaGetLastError();
	printf("%s: %s\n", cudaGetErrorName(err), cudaGetErrorString(err));
}

void cuda_main() {

	const int GRID = 10;
	const int BLOCK = 10;
	const int LS_SAMPLES = 5;

	/// COMPUTE LATIN SQUARES

	printf("Computing latin squares...\n");
	auto start_ls = clock();

	int* topology = new int[16 * 6];
	make_butterfly_butterfly_topology(topology);

	bool* char_mat = new bool[16 * 8 * 7];
	make_characteristic_matrices(0, char_mat);

	bool* dev_char_mat;
	int* dev_topology;
	bool* dev_conf;
	bool* dev_is_latin_square;
	int* dev_perm;
	curandState* dev_states1;

	CUDA_ERROR_CHECK(cudaMalloc((void**)&dev_char_mat, (16 * 8 * 7) * sizeof(bool)));
	CUDA_ERROR_CHECK(cudaMalloc((void**)&dev_topology, (16 * 6) * sizeof(int)));
	CUDA_ERROR_CHECK(cudaMalloc((void**)&dev_conf, (GRID * BLOCK) * (8 * 7) * sizeof(bool)));
	CUDA_ERROR_CHECK(cudaMalloc((void**)&dev_is_latin_square, (GRID * BLOCK) * sizeof(bool)));
	CUDA_ERROR_CHECK(cudaMalloc((void**)&dev_perm, (GRID * BLOCK) * (16 * 16) * sizeof(int)));
	CUDA_ERROR_CHECK(cudaMalloc((void**)&dev_states1, GRID * BLOCK * sizeof(curandState)));

	CUDA_ERROR_CHECK(cudaMemcpy(dev_char_mat, char_mat, (16 * 8 * 7) * sizeof(bool), cudaMemcpyHostToDevice));
	CUDA_ERROR_CHECK(cudaMemcpy(dev_topology, topology, (16 * 6) * sizeof(int), cudaMemcpyHostToDevice));

	setup_rand_state << <GRID, BLOCK >> > (dev_states1);
	cuda_handle_error();

	check_latin_square << <GRID, BLOCK >> > (dev_states1, dev_char_mat, dev_topology, dev_conf, dev_is_latin_square, dev_perm);
	cuda_handle_error();

	delete[] topology;
	delete[] char_mat;
	CUDA_ERROR_CHECK(cudaFree(dev_char_mat));
	CUDA_ERROR_CHECK(cudaFree(dev_topology));
	CUDA_ERROR_CHECK(cudaFree(dev_states1));

	bool* out_is_ls = new bool[GRID * BLOCK];
	bool* out_conf = new bool[(GRID * BLOCK) * (8 * 7)];
	int* out_perm = new int[(GRID * BLOCK) * (16 * 16)];

	CUDA_ERROR_CHECK(cudaMemcpy(out_is_ls, dev_is_latin_square, GRID * BLOCK * sizeof(bool), cudaMemcpyDeviceToHost));
	CUDA_ERROR_CHECK(cudaMemcpy(out_conf, dev_conf, (GRID * BLOCK) * (8 * 7) * sizeof(bool), cudaMemcpyDeviceToHost));
	CUDA_ERROR_CHECK(cudaMemcpy(out_perm, dev_perm, (GRID * BLOCK) * (16 * 16) * sizeof(int), cudaMemcpyDeviceToHost));

	CUDA_ERROR_CHECK(cudaFree(dev_conf));

	FILE* fd_ls = fopen(OUTFILE1, "w+");
	write_output_latin_square(fd_ls, out_is_ls, out_conf, out_perm, GRID, BLOCK);
	fclose(fd_ls);

	delete[] out_conf;

	printf("Done in %6.4f ms.\n", clock() - start_ls);

	/// END: COMPUTE LATIN SQUARES


	/// COMPUTE MUTUALLY ORTHOGONAL LATIN SQUARES

	printf("Computing MOLS...\n");
	auto start_mols = clock();

	bool* dev_mols;
	int* dev_pairs;
	curandState* dev_states2;
	CUDA_ERROR_CHECK(cudaMalloc((void**)&dev_states2, (GRID * BLOCK * LS_SAMPLES) * sizeof(curandState)));
	CUDA_ERROR_CHECK(cudaMalloc((void**)&dev_mols, (GRID * BLOCK * LS_SAMPLES) * sizeof(bool)));
	CUDA_ERROR_CHECK(cudaMalloc((void**)&dev_pairs, (GRID * BLOCK * LS_SAMPLES * 2) * sizeof(int)));

	auto grid_size = dim3(GRID, BLOCK, 1);

	setup_rand_state << <grid_size, LS_SAMPLES >> > (dev_states2);
	cuda_handle_error();

	check_mols << < grid_size, LS_SAMPLES >> > (dev_states2, dev_perm, dev_is_latin_square, dev_mols, dev_pairs);
	cuda_handle_error();

	bool* out_mols = new bool[GRID * BLOCK * LS_SAMPLES];
	int* out_pairs = new int[GRID * BLOCK * LS_SAMPLES * 2];

	CUDA_ERROR_CHECK(cudaMemcpy(out_mols, dev_mols, GRID * BLOCK * LS_SAMPLES * sizeof(bool), cudaMemcpyDeviceToHost));
	CUDA_ERROR_CHECK(cudaMemcpy(out_pairs, dev_pairs, GRID * BLOCK * LS_SAMPLES * 2 * sizeof(int), cudaMemcpyDeviceToHost));

	FILE* fd_mols = fopen(OUTFILE2, "w+");
	write_output_mols(fd_mols, out_mols, out_perm, out_pairs, GRID, BLOCK, LS_SAMPLES);
	fclose(fd_mols);

	printf("Done in %6.4f ms.\n", clock() - start_mols);

	/// END: COMPUTE MUTUALLY ORTHOGONAL LATIN SQUARES

	// release memory

	CUDA_ERROR_CHECK(cudaFree(dev_is_latin_square));
	CUDA_ERROR_CHECK(cudaFree(dev_perm));
	CUDA_ERROR_CHECK(cudaFree(dev_mols));
	CUDA_ERROR_CHECK(cudaFree(dev_pairs));
	CUDA_ERROR_CHECK(cudaFree(dev_states2));

	delete[] out_is_ls;
	delete[] out_perm;
	delete[] out_mols;
	delete[] out_pairs;

}
