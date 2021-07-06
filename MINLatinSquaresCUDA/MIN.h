#pragma once


void make_characteristic_matrices(unsigned int N, bool* out);

void make_butterfly_butterfly_topology(int* topology);

bool check_latin_square_sequential(bool* matrices, int* topology, bool* conf, int* perm);

void MIN_get_routes(int* topology, bool* conf, int* routes);
void MIN_modify_conf(int* topology, int* perm, bool* ROT);
void ID_rotation_configuration(int* topology, bool* ID, int rot_idx, bool* ID_rot);
void MIN_switch_identity(bool* ID_conf);
void generate_rotation_configurations(int* topology, bool* out);