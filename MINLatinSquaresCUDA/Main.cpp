#include "MINUtils.h"
#include <iostream>


int main() {

	bool* out = new bool[16 * 8 * 7];

	int* TOP = new int[16 * 7];
	generateButterflyButterflyTopology(TOP);
	for (int p = 0; p < 16; p++) {
		for (int st = 0; st < 7; st++) {
		
			std::cout << TOP[p*16+st] << ", ";
		}
		std::cout << std::endl;
	}
	/*generateCharacteristicMatrices(0, out);

	for (int m = 0; m < 16; m++) {
		std::cout << "Printing matrix " << m << std::endl;
		for (int i = 0; i < 8; i++) {
			for (int j = 0; j < 7; j++) {
				std::cout << out[(m * 56) + (i * 7) + j] << ", ";
			}
			std::cout << std::endl;
		}
	}*/

}