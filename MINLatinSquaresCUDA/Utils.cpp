#include "Utils.h"
#include <iostream>


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

