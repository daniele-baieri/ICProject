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

void write_output_latin_square(FILE* fd, bool* output, bool* out_conf, int* out_perm, int N, int M) {
	int idx;
	bool out_ij;
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < M; j++) {
			idx = i * M + j;
			out_ij = output[idx];
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

void write_output_mols(FILE* fd, bool* mols, int* perm, int* pair_idxs, int N, int M, int S) {
	int idx;
	bool out_ij;
	int pair_a, pair_b;
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < M; j++) {
			for (int k = 0; k < S; k++) {
				idx = (i * M + j) * S + k;
				out_ij = mols[idx];
				pair_a = pair_idxs[2 * idx];
				pair_b = pair_idxs[2 * idx + 1];
				fprintf(fd, "THREAD %d: OUTPUT = %d -- LATIN SQUARES = (%d, %d)\n\n", idx, out_ij, pair_a, pair_b);
				for (int x = 0; x < 16; x++) {
					for (int y = 0; y < 16; y++)
						fprintf(fd, "(%2d, %2d) ", perm[pair_a * 16 * 16 + x * 16 + y], perm[pair_b * 8 * 7 + x * 7 + y]);  // show mols by columns
					fprintf(fd, "\n");
				}
				fprintf(fd, "\n%s\n\n", DELIMITER);
			}
		}
	}
}