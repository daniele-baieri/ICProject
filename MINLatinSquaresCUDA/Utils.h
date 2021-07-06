#include <stdio.h>

#pragma once


void print_int_matrix(int* matrix, int n, int m);

void print_bool_matrix(bool* matrix, int n, int m);

void write_output_latin_square(FILE* fd, bool* output, bool* out_conf, int* out_perm, int N, int M);

void write_output_mols(FILE* fd, bool* mols, int* perm, int* pairs, int N, int M, int S);

bool get_bit(int n, int pos, int tot_bit);