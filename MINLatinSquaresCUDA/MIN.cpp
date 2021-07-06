#include "MIN.h"
#include "Utils.h"
#include <cstdlib>


void make_characteristic_matrices(unsigned int N, bool* out) {
    // auto N = 16;
    bool row[7];
    /*
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
    */
    for (int b_3 = 0; b_3 <= 1; b_3++) {
        for (int b_4 = 0; b_4 <= 1; b_4++) {
            for (int b_5 = 0; b_5 <= 1; b_5++) {
                for (int b_6 = 0; b_6 <= 1; b_6++) {
                    row[3] = b_3;
                    row[4] = b_4;
                    row[5] = b_5;
                    row[6] = b_6;
                    int idx = 8 * b_3 + 4 * b_4 + 2 * b_5 + 1 * b_6;
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

void make_butterfly_butterfly_topology(int* topology) {
    // topology(p,st) = arrival port at stage st+1 of outgoing link at port p of stage st
    for (int st = 0; st < 6; st++) {
        for (int sw = 0; sw < 8; sw++) {
            // printf("Considering stage %d and starting switch %d\n", st, sw);
            // selecting upper and lower port of switch sw at stage st
            int port_up = sw * 2;
            int port_down = sw * 2 + 1;
            //printf("starting ports: %d, %d\n", port_up, port_down);
            // straight edge goes to switch sw at stage st+1
            int straight_sw = sw;
            //printf("straight switch: %d\n", straight_sw);
            // cross edge goes to switch sw' at stage st+1 where sw' = sw, but the bit (st mod 4) is complemented
            int bit_pos = 3 - 1 - st % 3;
            int bit_st = ((sw & (1 << bit_pos)) >> bit_pos);
            //printf("bit n %d: %d\n", bit_pos, bit_st);
            int mask = 1 << bit_pos;
            int cross_sw = ((sw & ~mask) | (!bit_st << bit_pos));

            //printf("cross switch: %d\n", cross_sw);
            // find the arrival ports
            int straight_port, cross_port_depart, cross_port_arrival;
            if (bit_st == 0) {
                // upper half of subnetwork
                // upper edges are straight both depart and arrival
                straight_port = port_up;
                // lower edges are cross at depart and upper at arrival
                cross_port_depart = port_down;
                cross_port_arrival = cross_sw * 2; // upper port of arrival sw
            }
            else {
                // lower half of subnetwork
                // lower edges are straight both depart and arrival
                straight_port = port_down;
                // upper edges are cross at depart and lower at arrival
                cross_port_depart = port_up;
                cross_port_arrival = cross_sw * 2 + 1; // lower port of arrival sw
            }
            topology[straight_port * 6 + st] = straight_port;
            topology[cross_port_depart * 6 + st] = cross_port_arrival;
            //printf("depart port: %d, arrival port: %d\n", straight_port, straight_port);
            // printf("from %d we go in %d at stage %d\n", straight_port, topology[straight_port * 6 + st], st);
            //printf("depart port: %d, arrival port: %d\n", cross_port_depart, cross_port_arrival);
            // printf("from %d we go in %d at stage %d\n", cross_port_depart, topology[cross_port_depart * 6 + st], st);


        }
    }
}

bool check_latin_square_sequential(bool* matrices, int* topology, bool* conf, int* perm) {

    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 7; j++) {
            conf[i * 7 + j] = (rand() >= 0.5f ? true : false);
        }
    }

    bool xor [8 * 7];
    int curr_perm[16 * 7];
    for (int m = 0; m < 16; m++) {
        bool* to_xor = &matrices[m * 8 * 7];
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 7; j++) {
                auto idx = i * 7 + j;
                xor [idx] = (!conf[idx] != !to_xor[idx]);
            }
        }

        for (int stage = 0; stage < 7; stage++) {
            for (int sw = 0; sw < 8; sw++) {
                // apply switches
                int up_port = sw * 2;
                int low_port = sw * 2 + 1;
                if (stage == 0) {
                    curr_perm[up_port * 7 + stage] = up_port;
                    curr_perm[low_port * 7 + stage] = low_port;
                }

                bool setting = xor [sw * 7 + stage];
                if (setting) {
                    int lp = curr_perm[low_port * 7 + stage];
                    curr_perm[low_port * 7 + stage] = curr_perm[up_port * 7 + stage];
                    curr_perm[up_port * 7 + stage] = lp;
                }
                if (stage == 6) {
                    perm[m * 16 + up_port] = curr_perm[up_port * 7 + stage];
                    perm[m * 16 + low_port] = curr_perm[low_port * 7 + stage];
                }

            }
            for (int port = 0; port < 16 && stage < 6; port++) {
                // apply topology
                curr_perm[topology[port * 6 + stage] * 7 + (stage + 1)] = curr_perm[port * 7 + stage];
            }
        }


    }

    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            int v = perm[i * 16 + j];
            for (int r = i; r < 16; r++) {
                if (perm[r * 16 + j] == v) return false;
            }
        }
    }

    return true;

}
