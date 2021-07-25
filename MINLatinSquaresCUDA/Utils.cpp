#include "Utils.h"
#include <iostream>

#define DELIMITER "-----------"


void print_int_matrix(int* matrix, int n, int m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            printf("%d, ", matrix[i * m + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void print_bool_matrix(bool* matrix, int n, int m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            printf("%d, ", matrix[i * m + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void write_output_latin_square(FILE* fd, bool* output, bool* out_conf, int* out_perm, int N) {
	bool out_ij;
	for (int idx = 0; idx < N; idx++) {
		out_ij = output[idx];
		if (out_ij || true) {  // write all output for latin squares
			fprintf(fd, "THREAD %d: OUTPUT = %d\n\n", idx, out_ij);
			for (int x = 0; x < 16; x++) {
				for (int y = 0; y < 16; y++)
					fprintf(fd, "%2d ", out_perm[idx * 16 * 16 + x * 16 + y]);  // show resulting permutations by columns
				fprintf(fd, "| ");
				for (int y = 0; y < 7 && x < 8; y++)
					fprintf(fd, "%d ", out_conf[idx * 8 * 7 + x * 7 + y]);  // show random configuration on the side
				fprintf(fd, "\n");
			}
			fprintf(fd, "\n%s\n\n", DELIMITER);
		}
	}
}

void write_output_mols(FILE* fd, bool* mols, int* perm, int* pair_idxs, int N) {
	bool out_ij;
	int pair_a, pair_b;
	for (int i = 0; i < N; i++) {
		out_ij = mols[i];
		if (out_ij) {  // write only mutually orthogonal LS
			pair_a = pair_idxs[2 * i];
			pair_b = pair_idxs[2 * i + 1];
			fprintf(fd, "THREAD %d: OUTPUT = %d -- LATIN SQUARES = (%d, %d)\n\n", i, out_ij, pair_a, pair_b);
			for (int x = 0; x < 16; x++) {
				for (int y = 0; y < 16; y++)
					fprintf(fd, "(%2d, %2d) ", perm[pair_a * 16 * 16 + x * 16 + y], perm[pair_b * 16 * 16 + x * 16 + y]);  // show mols by columns
				fprintf(fd, "\n");
			}
			fprintf(fd, "\n%s\n\n", DELIMITER);
		}
	}
}


bool get_bit(int n, int pos, int tot_bit) {
    /*
    return bit in position pos starting from left
    */
    bool remainder;
    int n_div = tot_bit - pos;
    while (n_div > 0) {
        remainder = n % 2;
        n = n / 2;
        n_div--;
    }
    return remainder;
}