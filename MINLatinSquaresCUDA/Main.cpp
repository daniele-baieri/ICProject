#include "MINUtils.h"
#include <iostream>
#include "LatinSquares.cuh"
#include <cstdlib>


void testLatinSquare(bool* matrices, int* topology, bool* conf) {

	for (int i = 0; i < 8; i++) {
		for (int j = 0; j < 7; j++) {
			conf[i * 7 + j] = rand() & 1;
		}
	}

	printBoolMatrix(conf, 8, 7);

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

		if (m == 0) {
			printBoolMatrix(to_xor, 8, 7);
			printBoolMatrix(xor, 8, 7);
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

		if (m == 0) printIntMatrix(curr_perm, 16, 7);

	}

}



int main() {

	int* TOP = new int[16 * 6];
	generateButterflyButterflyTopology(TOP);
	printIntMatrix(TOP, 16, 6);

	bool* out = new bool[16 * 8 * 7];
	generateCharacteristicMatrices(0, out);

	bool* conf = new bool[8 * 7];
	testLatinSquare(out, TOP, conf);

}