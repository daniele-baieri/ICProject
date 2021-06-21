#include "MINUtils.h"
#include <iostream>


int main() {

	bool* out = new bool[16 * 8 * 7];
	generateCharacteristicMatrices(0, out);

	for (int m = 0; m < 16; m++) {
		std::cout << "Printing matrix " << m << std::endl;
		for (int i = 0; i < 8; i++) {
			for (int j = 0; j < 7; j++) {
				std::cout << out[(m * 56) + (i * 7) + j] << ", ";
			}
			std::cout << std::endl;
		}
	}

}