#include "LatinSquares.cuh"
#include "Utils.h"
#include "MIN.h"
#include <math.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <time.h>


constexpr auto OUTFILE1 = "./Results/results-ls.txt";
constexpr auto OUTFILE2 = "./Results/results-mols.txt";
constexpr auto DEBUG_LOG = "./Results/log.txt";
constexpr auto CROSS_SEARCH = true;


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

void cross_method_search() {

	const unsigned int N = 16;
	const unsigned int SWITCHES = N / 2;
	const unsigned int STAGES = (2 * log2(N)) - 1;

	const unsigned int MOLS_GRID_PARALLEL = 200;
	const unsigned int MOLS_SEQ_CUTOFF = 100;
	const unsigned int NUM_LATIN_SQUARES = MOLS_GRID_PARALLEL * MOLS_SEQ_CUTOFF;

	const bool DEBUG = false;

	freopen(DEBUG_LOG, "w+", stdout);

	/// COMPUTE LATIN SQUARES

	printf("Computing latin squares...\n");
	auto start_ls = clock();

	int* topology = new int[16 * 6];
	make_butterfly_butterfly_topology(topology);

	bool* char_mat = new bool[16 * 8 * 7];
	bool* rot_mat = new bool[16 * 8 * 7];
	make_characteristic_matrices(0, char_mat);
	generate_rotation_configurations(topology, rot_mat);


	bool* dev_char_mat;
	bool* dev_rot_mat;

	int* dev_topology;

	bool* dev_conf_char;
	bool* dev_conf_rot;
	bool* dev_is_latin_square_char;
	bool* dev_is_latin_square_rot;
	int* dev_perm_char;
	int* dev_perm_rot;
	curandState* dev_states1;

	CUDA_ERROR_CHECK(cudaMalloc((void**)&dev_char_mat, (16 * 8 * 7) * sizeof(bool)));
	CUDA_ERROR_CHECK(cudaMalloc((void**)&dev_rot_mat, (16 * 8 * 7) * sizeof(bool)));
	CUDA_ERROR_CHECK(cudaMalloc((void**)&dev_topology, (16 * 6) * sizeof(int)));
	CUDA_ERROR_CHECK(cudaMalloc((void**)&dev_conf_char, (NUM_LATIN_SQUARES) * (8 * 7) * sizeof(bool)));
	CUDA_ERROR_CHECK(cudaMalloc((void**)&dev_is_latin_square_char, (NUM_LATIN_SQUARES) * sizeof(bool)));
	CUDA_ERROR_CHECK(cudaMalloc((void**)&dev_perm_char, (NUM_LATIN_SQUARES) * (16 * 16) * sizeof(int)));
	CUDA_ERROR_CHECK(cudaMalloc((void**)&dev_conf_rot, (NUM_LATIN_SQUARES) * (8 * 7) * sizeof(bool)));
	CUDA_ERROR_CHECK(cudaMalloc((void**)&dev_is_latin_square_rot, (NUM_LATIN_SQUARES) * sizeof(bool)));
	CUDA_ERROR_CHECK(cudaMalloc((void**)&dev_perm_rot, (NUM_LATIN_SQUARES) * (16 * 16) * sizeof(int)));
	CUDA_ERROR_CHECK(cudaMalloc((void**)&dev_states1, NUM_LATIN_SQUARES * sizeof(curandState)));

	CUDA_ERROR_CHECK(cudaMemcpy(dev_char_mat, char_mat, (16 * 8 * 7) * sizeof(bool), cudaMemcpyHostToDevice));
	CUDA_ERROR_CHECK(cudaMemcpy(dev_rot_mat, rot_mat, (16 * 8 * 7) * sizeof(bool), cudaMemcpyHostToDevice));
	CUDA_ERROR_CHECK(cudaMemcpy(dev_topology, topology, (16 * 6) * sizeof(int), cudaMemcpyHostToDevice));

	setup_rand_state << <NUM_LATIN_SQUARES, 1 >> > (dev_states1);
	cuda_handle_error();

	check_latin_square << <NUM_LATIN_SQUARES, 1 >> > (
		dev_states1, dev_char_mat, dev_topology, dev_conf_char, dev_is_latin_square_char, dev_perm_char
	);
	cuda_handle_error();

	check_latin_square << <NUM_LATIN_SQUARES, 1 >> > (
		dev_states1, dev_rot_mat, dev_topology, dev_conf_rot, dev_is_latin_square_rot, dev_perm_rot
	);
	cuda_handle_error();

	delete[] topology;
	delete[] char_mat;
	delete[] rot_mat;
	CUDA_ERROR_CHECK(cudaFree(dev_char_mat));
	CUDA_ERROR_CHECK(cudaFree(dev_rot_mat));
	CUDA_ERROR_CHECK(cudaFree(dev_topology));
	CUDA_ERROR_CHECK(cudaFree(dev_states1));

	bool* out_is_ls_char = new bool[NUM_LATIN_SQUARES];
	bool* out_is_ls_rot = new bool[NUM_LATIN_SQUARES];
	bool* out_conf_char = new bool[(NUM_LATIN_SQUARES) * (8 * 7)];
	bool* out_conf_rot = new bool[(NUM_LATIN_SQUARES) * (8 * 7)];
	int* out_perm_char = new int[(NUM_LATIN_SQUARES) * (16 * 16)];
	int* out_perm_rot = new int[(NUM_LATIN_SQUARES) * (16 * 16)];

	CUDA_ERROR_CHECK(cudaMemcpy(out_is_ls_char, dev_is_latin_square_char, NUM_LATIN_SQUARES * sizeof(bool), cudaMemcpyDeviceToHost));
	CUDA_ERROR_CHECK(cudaMemcpy(out_conf_char, dev_conf_char, (NUM_LATIN_SQUARES) * (8 * 7) * sizeof(bool), cudaMemcpyDeviceToHost));
	CUDA_ERROR_CHECK(cudaMemcpy(out_perm_char, dev_perm_char, (NUM_LATIN_SQUARES) * (16 * 16) * sizeof(int), cudaMemcpyDeviceToHost));
	CUDA_ERROR_CHECK(cudaMemcpy(out_is_ls_rot, dev_is_latin_square_rot, NUM_LATIN_SQUARES * sizeof(bool), cudaMemcpyDeviceToHost));
	CUDA_ERROR_CHECK(cudaMemcpy(out_conf_rot, dev_conf_rot, (NUM_LATIN_SQUARES) * (8 * 7) * sizeof(bool), cudaMemcpyDeviceToHost));
	CUDA_ERROR_CHECK(cudaMemcpy(out_perm_rot, dev_perm_rot, (NUM_LATIN_SQUARES) * (16 * 16) * sizeof(int), cudaMemcpyDeviceToHost));

	CUDA_ERROR_CHECK(cudaFree(dev_conf_char));
	CUDA_ERROR_CHECK(cudaFree(dev_conf_rot));

	FILE* fd_ls = fopen(OUTFILE1, "w+");
	fprintf(fd_ls, "Latin squares for characteristic matrices:\n\n");
	write_output_latin_square(fd_ls, out_is_ls_char, out_conf_char, out_perm_char, NUM_LATIN_SQUARES);
	fprintf(fd_ls, "Latin squares for rotation matrices:\n\n");
	write_output_latin_square(fd_ls, out_is_ls_rot, out_conf_rot, out_perm_rot, NUM_LATIN_SQUARES);
	fclose(fd_ls);

	delete[] out_conf_char;
	delete[] out_conf_rot;
	delete[] out_is_ls_char;
	delete[] out_is_ls_rot;

	auto end_ls = clock();
	printf("Done in %6.4f ms.\n", (double)(end_ls - start_ls) / CLOCKS_PER_SEC);

	/// END: COMPUTE LATIN SQUARES

	
	/// COMPUTE MUTUALLY ORTHOGONAL LATIN SQUARES

	printf("Computing MOLS...\n");
	auto start_mols = clock();

	bool* dev_mols;
	int* dev_pairs;
	curandState* dev_states2;
	dim3 grid_size;
	unsigned int NUM_COMPARISONS = NUM_LATIN_SQUARES * NUM_LATIN_SQUARES;

	CUDA_ERROR_CHECK(cudaMalloc((void**)&dev_mols, (NUM_COMPARISONS) * sizeof(bool)));
	CUDA_ERROR_CHECK(cudaMalloc((void**)&dev_pairs, (NUM_COMPARISONS * 2) * sizeof(int)));
	grid_size = dim3(NUM_LATIN_SQUARES, MOLS_GRID_PARALLEL, 1);
	check_mols_complete << < grid_size, MOLS_SEQ_CUTOFF >> > (
		dev_perm_char, dev_perm_rot, dev_is_latin_square_char, dev_is_latin_square_rot, dev_mols, dev_pairs, DEBUG
	);
	cuda_handle_error();

	CUDA_ERROR_CHECK(cudaFree(dev_is_latin_square_char))
	CUDA_ERROR_CHECK(cudaFree(dev_is_latin_square_rot));
	CUDA_ERROR_CHECK(cudaFree(dev_perm_char));
	CUDA_ERROR_CHECK(cudaFree(dev_perm_rot));

	bool* out_mols = new bool[NUM_COMPARISONS];
	int* out_pairs = new int[NUM_COMPARISONS * 2];

	CUDA_ERROR_CHECK(cudaMemcpy(out_mols, dev_mols, NUM_COMPARISONS * sizeof(bool), cudaMemcpyDeviceToHost));
	CUDA_ERROR_CHECK(cudaMemcpy(out_pairs, dev_pairs, NUM_COMPARISONS * 2 * sizeof(int), cudaMemcpyDeviceToHost));
	CUDA_ERROR_CHECK(cudaFree(dev_mols));
	CUDA_ERROR_CHECK(cudaFree(dev_pairs));

	FILE* fd_mols = fopen(OUTFILE2, "w+");
	write_output_mols(fd_mols, out_mols, out_perm_char, out_perm_rot, out_pairs, NUM_COMPARISONS);
	fclose(fd_mols);

	auto end_mols = clock();
	printf("Done in %6.4f ms.\n", (double)(end_mols - start_mols) / CLOCKS_PER_SEC);



	/// END: COMPUTE MUTUALLY ORTHOGONAL LATIN SQUARES

	// release memory

	delete[] out_perm_char;
	delete[] out_perm_rot;
	delete[] out_mols;
	delete[] out_pairs;
	
}

void single_method_search() {

	const unsigned int METHOD = 0;

	const unsigned int N = 16;
	const unsigned int SWITCHES = N / 2;
	const unsigned int STAGES = (2 * log2(N)) - 1;

	const unsigned int MOLS_GRID_PARALLEL = 10;
	const unsigned int MOLS_SEQ_CUTOFF = 8;
	const unsigned int NUM_LATIN_SQUARES = MOLS_GRID_PARALLEL * MOLS_SEQ_CUTOFF;
	// const unsigned int BLOCK = 10;

	const unsigned int LS_SAMPLES = 200;
	const bool DO_COMPLETE_CHECK = true;
	const bool DEBUG = true;

	freopen(DEBUG_LOG, "w+", stdout);

	/// COMPUTE LATIN SQUARES

	printf("Computing latin squares...\n");
	auto start_ls = clock();

	int* topology = new int[16 * 6];
	make_butterfly_butterfly_topology(topology);

	bool* char_mat = new bool[16 * 8 * 7];
	switch (METHOD) {
	case 0: make_characteristic_matrices(0, char_mat); break;
	case 1: generate_rotation_configurations(topology, char_mat); break;
	}

	bool* dev_char_mat;
	int* dev_topology;
	bool* dev_conf;
	bool* dev_is_latin_square;
	int* dev_perm;
	curandState* dev_states1;

	CUDA_ERROR_CHECK(cudaMalloc((void**)&dev_char_mat, (16 * 8 * 7) * sizeof(bool)));
	CUDA_ERROR_CHECK(cudaMalloc((void**)&dev_topology, (16 * 6) * sizeof(int)));
	CUDA_ERROR_CHECK(cudaMalloc((void**)&dev_conf, (NUM_LATIN_SQUARES) * (8 * 7) * sizeof(bool)));
	CUDA_ERROR_CHECK(cudaMalloc((void**)&dev_is_latin_square, (NUM_LATIN_SQUARES) * sizeof(bool)));
	CUDA_ERROR_CHECK(cudaMalloc((void**)&dev_perm, (NUM_LATIN_SQUARES) * (16 * 16) * sizeof(int)));
	CUDA_ERROR_CHECK(cudaMalloc((void**)&dev_states1, NUM_LATIN_SQUARES * sizeof(curandState)));

	CUDA_ERROR_CHECK(cudaMemcpy(dev_char_mat, char_mat, (16 * 8 * 7) * sizeof(bool), cudaMemcpyHostToDevice));
	CUDA_ERROR_CHECK(cudaMemcpy(dev_topology, topology, (16 * 6) * sizeof(int), cudaMemcpyHostToDevice));

	setup_rand_state << <NUM_LATIN_SQUARES, 1 >> > (dev_states1);
	cuda_handle_error();

	check_latin_square << <NUM_LATIN_SQUARES, 1 >> > (dev_states1, dev_char_mat, dev_topology, dev_conf, dev_is_latin_square, dev_perm);
	cuda_handle_error();

	delete[] topology;
	delete[] char_mat;
	CUDA_ERROR_CHECK(cudaFree(dev_char_mat));
	CUDA_ERROR_CHECK(cudaFree(dev_topology));
	CUDA_ERROR_CHECK(cudaFree(dev_states1));

	bool* out_is_ls = new bool[NUM_LATIN_SQUARES];
	bool* out_conf = new bool[(NUM_LATIN_SQUARES) * (8 * 7)];
	int* out_perm = new int[(NUM_LATIN_SQUARES) * (16 * 16)];

	CUDA_ERROR_CHECK(cudaMemcpy(out_is_ls, dev_is_latin_square, NUM_LATIN_SQUARES * sizeof(bool), cudaMemcpyDeviceToHost));
	CUDA_ERROR_CHECK(cudaMemcpy(out_conf, dev_conf, (NUM_LATIN_SQUARES) * (8 * 7) * sizeof(bool), cudaMemcpyDeviceToHost));
	CUDA_ERROR_CHECK(cudaMemcpy(out_perm, dev_perm, (NUM_LATIN_SQUARES) * (16 * 16) * sizeof(int), cudaMemcpyDeviceToHost));

	CUDA_ERROR_CHECK(cudaFree(dev_conf));

	FILE* fd_ls = fopen(OUTFILE1, "w+");
	write_output_latin_square(fd_ls, out_is_ls, out_conf, out_perm, NUM_LATIN_SQUARES);
	fclose(fd_ls);

	delete[] out_conf;
	delete[] out_is_ls;

	auto end_ls = clock();
	printf("Done in %6.4f ms.\n", (double)(end_ls - start_ls) / CLOCKS_PER_SEC);

	/// END: COMPUTE LATIN SQUARES


	/// COMPUTE MUTUALLY ORTHOGONAL LATIN SQUARES

	printf("Computing MOLS, complete check: %d...\n", DO_COMPLETE_CHECK);
	auto start_mols = clock();

	bool* dev_mols;
	int* dev_pairs;
	curandState* dev_states2;
	dim3 grid_size;
	unsigned int NUM_COMPARISONS;

	if (DO_COMPLETE_CHECK) {
		NUM_COMPARISONS = NUM_LATIN_SQUARES * NUM_LATIN_SQUARES;
		CUDA_ERROR_CHECK(cudaMalloc((void**)&dev_mols, (NUM_COMPARISONS) * sizeof(bool)));
		CUDA_ERROR_CHECK(cudaMalloc((void**)&dev_pairs, (NUM_COMPARISONS * 2) * sizeof(int)));
		grid_size = dim3(NUM_LATIN_SQUARES, MOLS_GRID_PARALLEL, 1);
		check_mols_complete << < grid_size, MOLS_SEQ_CUTOFF >> > (dev_perm, dev_is_latin_square, dev_mols, dev_pairs, DEBUG);
		cuda_handle_error();
		CUDA_ERROR_CHECK(cudaDeviceSynchronize());
	}
	else {
		NUM_COMPARISONS = NUM_LATIN_SQUARES * LS_SAMPLES;
		CUDA_ERROR_CHECK(cudaMalloc((void**)&dev_states2, (NUM_COMPARISONS) * sizeof(curandState)));
		CUDA_ERROR_CHECK(cudaMalloc((void**)&dev_mols, (NUM_COMPARISONS) * sizeof(bool)));
		CUDA_ERROR_CHECK(cudaMalloc((void**)&dev_pairs, (NUM_COMPARISONS * 2) * sizeof(int)));
		setup_rand_state << <NUM_LATIN_SQUARES, LS_SAMPLES >> > (dev_states2);
		cuda_handle_error();
		check_mols_random << < NUM_LATIN_SQUARES, LS_SAMPLES >> > (dev_states2, dev_perm, dev_is_latin_square, dev_mols, dev_pairs, DEBUG);
		cuda_handle_error();
		CUDA_ERROR_CHECK(cudaFree(dev_states2));
	}

	CUDA_ERROR_CHECK(cudaFree(dev_is_latin_square));
	CUDA_ERROR_CHECK(cudaFree(dev_perm));

	bool* out_mols = new bool[NUM_COMPARISONS];
	int* out_pairs = new int[NUM_COMPARISONS * 2];

	CUDA_ERROR_CHECK(cudaMemcpy(out_mols, dev_mols, NUM_COMPARISONS * sizeof(bool), cudaMemcpyDeviceToHost));
	CUDA_ERROR_CHECK(cudaMemcpy(out_pairs, dev_pairs, NUM_COMPARISONS * 2 * sizeof(int), cudaMemcpyDeviceToHost));
	CUDA_ERROR_CHECK(cudaFree(dev_mols));
	CUDA_ERROR_CHECK(cudaFree(dev_pairs));

	FILE* fd_mols = fopen(OUTFILE2, "w+");
	write_output_mols(fd_mols, out_mols, out_perm, out_pairs, NUM_COMPARISONS);
	fclose(fd_mols);

	auto end_mols = clock();
	printf("Done in %6.4f ms.\n", (double)(end_mols - start_mols) / CLOCKS_PER_SEC);



	/// END: COMPUTE MUTUALLY ORTHOGONAL LATIN SQUARES

	// release memory

	delete[] out_perm;
	delete[] out_mols;
	delete[] out_pairs;

}

void cuda_main() {

	if (CROSS_SEARCH) {
		cross_method_search();
	}
	else {
		single_method_search();
	}

}
