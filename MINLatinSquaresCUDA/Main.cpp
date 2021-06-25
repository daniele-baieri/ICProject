#include "MINUtils.h"
#include <iostream>
#include "LatinSquares.cuh"
#include <cstdlib>


bool test_latin_square(bool* matrices, int* topology, bool* conf) {

	for (int i = 0; i < 8; i++) {
		for (int j = 0; j < 7; j++) {
			conf[i * 7 + j] = rand() & 1;
		}
	}

	print_bool_matrix(conf, 8, 7);

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

		if (m == 1) {
			print_bool_matrix(to_xor, 8, 7);
			print_bool_matrix(xor, 8, 7);
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
					perm[m * 16 + up_port] = curr_perm[up_port * 7 + stage];
					perm[m * 16 + low_port] = curr_perm[low_port * 7 + stage];
				}

			}
			for (int port = 0; port < 16 && stage < 6; port++) {
				// apply topology
				curr_perm[topology[port * 6 + stage] * 7 + (stage + 1)] = curr_perm[port * 7 + stage];
			}
		}

		if (m == 1) print_int_matrix(curr_perm, 16, 7);

	}

	print_int_matrix(perm, 16, 16);

	for (int i = 0; i < 16; i++) {
		for (int j = 0; j < 16; j++) {
			int v = perm[i * 16 + j];
			for (int r = i; r < 16; r++) {
				if (perm[r * 16 + j] == v) return false;
			}
		}
	}

	return true;

}


extern void cuda_main();

int main() {

	cuda_main();

}