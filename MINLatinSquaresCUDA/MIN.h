#pragma once


void make_characteristic_matrices(unsigned int N, bool* out);

void make_butterfly_butterfly_topology(int* topology);

bool check_latin_square_sequential(bool* matrices, int* topology, bool* conf, int* perm);