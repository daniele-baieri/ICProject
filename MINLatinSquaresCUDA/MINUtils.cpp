#include <algorithm>
#include <iterator>

#include "MINUtils.h"


void generateCharacteristicMatrices(unsigned int N, bool* out) {
	// auto N = 16;
	bool row[7];
	for (int b_0 = 0; b_0 <= 1; b_0++) {
		for (int b_1 = 0; b_1 <= 1; b_1++) {
			for (int b_3 = 0; b_3 <= 1; b_3++) {
				for (int b_6 = 0; b_6 <= 1; b_6++) {
					row[0] = b_0;
					row[1] = b_1;
					row[3] = b_3;
					row[6] = b_6;
					int idx = 8 * b_0 + 4 * b_1 + 2 * b_3 + 1 * b_6;
					for (int i = 0; i < 8; i++) {
						for (int j = 0; j < 7; j++) {
							out[(idx * 8 * 7) + (i * 7) + j] = row[j];
						}
					}
				}
			}
		}
	}
}