#include "MIN.h"
#include "Utils.h"


extern void cuda_main();

int main() {

	bool sequential = false;

	if (sequential) {

		int* topology = new int[16 * 6];
		make_butterfly_butterfly_topology(topology);

		bool* char_mat = new bool[16 * 8 * 7];
		make_characteristic_matrices(0, char_mat);
		print_bool_matrix(char_mat, 8, 7);

		int* perm = new int[16 * 16];
		bool conf[] = { 
			0, 0, 1, 0, 1, 1, 0,
			1, 0, 0, 1, 1, 1, 1,
			1, 1, 0, 0, 0, 1, 1,
			0, 0, 1, 0, 1, 0, 1,
			0, 1, 0, 0, 1, 0, 0,
			1, 1, 0, 1, 1, 1, 0,
			1, 1, 1, 1, 0, 1, 1,
			1, 1, 0, 1, 1, 1, 0
		};
		check_latin_square_sequential(char_mat, topology, conf, perm);
		print_int_matrix(perm, 16, 16);


	} else {

		cuda_main();

	}


}