#include "LatinSquares.cuh"
#include <cstdlib>


__global__ void checkLatinSquare(bool* matrices, int* topology) {
	bool conf[8 * 7];
	for (int i = 0; i < 8; i++) {
		for (int j = 0; j < 7; j++) {
			conf[i * 7 + j] = rand() & 1;
		}
	}

	bool perm[16 * 16];
	bool xor[8 * 7];
	for (int m = 0; m < 16; m++) {
		bool* to_xor = &matrices[m * 8 * 7];
		for (int i = 0; i < 8; i++) {
			for (int j = 0; j < 7; j++) {
				auto idx = i * 7 + j;
				xor [idx] = (!conf[idx] != !to_xor[idx]);
			}
		}
		for (int p = 0; p < 16; p++) {
			int swtch = (int)(p / 2);
			for (int stage = 0; stage < 7; stage++) {
				bool link = xor [swtch * 7 + stage];
				bool stage_bit = (swtch >> stage) & 1;
			}
			
		}
	}

}