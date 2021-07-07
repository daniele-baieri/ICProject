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
                    row[2] = 0;
                    row[3] = b_3;
                    row[4] = 0;
                    row[5] = 0;
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
                    row[0] = 0;
                    row[1] = 0;
                    row[2] = 0;
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

    /*
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 7; j++) {
            conf[i * 7 + j] = (rand() >= 0.5f ? true : false);
        }
    }
    */

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


void MIN_get_routes(int* topology, bool* conf, int* routes) {
    /*
      INPUT: 
        int* topology <- describe BB links
        bool* conf <- switch configuration
      OUTPUT:
        int* routes <- for each stage we represent only the output column (DIM: N_input x n_stages)
    */
    const unsigned int N_input = 16;
    const unsigned int n_stages = 7;
    unsigned int n_switches = N_input / 2;

    for (int stage = 0; stage < n_stages; stage++) {
        for (int sw = 0; sw < n_switches; sw++) {
            // apply switches
            int up_port = sw * 2;
            int low_port = sw * 2 + 1;
            if (stage == 0) {
                routes[up_port * n_stages + stage] = up_port;
                routes[low_port * n_stages + stage] = low_port;
            }

            bool setting = conf[sw * n_stages + stage];
            if (setting) {
                int lp = routes[low_port * n_stages + stage];
                routes[low_port * n_stages + stage] = routes[up_port * n_stages + stage];
                routes[up_port * n_stages + stage] = lp;
            }

        }
        for (int port = 0; port < N_input && stage < n_stages-1; port++) {
            // apply topology
            routes[topology[port * (n_stages-1) + stage] * n_stages + (stage + 1)] = routes[port * n_stages + stage];
        }
    }
}

void MIN_modify_conf(int* topology, int* perm, bool* ROT) {
    /*
      INPUT:
        int* topology <- describe BB links
        int* perm <- permutation for middle stage output
      OUTPUT:
        
        bool* ROT <- switch configuration dim 8*7 = N_nodes * n_stages
    */
    
    unsigned int N_input = 16;
    unsigned int n_stagesHalf = 4;//log(N_input)
    unsigned int n_stages = 2 * n_stagesHalf - 1;
    unsigned int N_nodes = N_input / 2;

    int* route = new int[N_input * 2 * n_stagesHalf];
    int* middle_position = new int[N_input];
    for (int input = 0; input < N_input; input++) {
        middle_position[perm[input]] = input;
        route[input * 2 * n_stagesHalf] = input;
    }
    for (int stage = 0; stage < n_stagesHalf; stage++){
        for (int port = 0; port < N_input; port = port + 2){            
            int x = middle_position[route[port * 2 * n_stagesHalf + 2 * stage]];
            bool bit = get_bit(x, stage, n_stagesHalf);
            if (bit) {
                ROT[int(ceil(port / 2)) * n_stages + stage] = 1;
                route[port * 2 * n_stagesHalf + 2 * stage+1] = route[(port + 1) * 2 * n_stagesHalf + 2 * stage];
                route[(port + 1) * 2 * n_stagesHalf + 2 * stage+1] = route[port * 2 * n_stagesHalf + 2 * stage];
            }
            else{
                ROT[int(ceil(port / 2)) * n_stages + stage] = 0;
                route[port * 2 * n_stagesHalf + 2 * stage+1] = route[port * 2 * n_stagesHalf + 2 * stage];
                route[(port + 1) * 2 * n_stagesHalf + 2 * stage+1] = route[(port + 1) * 2 * n_stagesHalf + 2 * stage];
            }            
        }
        for (int port = 0; port < N_input && stage < n_stagesHalf - 1; port++) {
            // apply topology
            int dest_port = topology[port * (n_stages - 1) + stage];
            route[ dest_port * 2 * n_stagesHalf + (2*stage + 2)] = route[port * 2 * n_stagesHalf + (2*stage+1)];
        }
    } 
    
}

void ID_rotation_configuration(int* topology, bool* ID, int rot_idx, bool* ID_rot) {
    unsigned int N_input = 16;
    unsigned int n_stagesHalf = 4;//log(N_input)
    unsigned int n_stages = 2 * n_stagesHalf - 1;
    unsigned int N_nodes = N_input / 2;

    int* routes = new int[16 * 7];
    int* output_middle_stage = new int[16];
    int* new_perm = new int[16];



    if ( rot_idx % 2 == 0) {
        for (int stage = 0; stage < n_stages; stage++) {
            for (int sw = 0; sw < N_nodes; sw++) {
                ID_rot[sw * n_stages + stage] = ID[sw * n_stages + stage];
            }
        }
        // from ID get output permutation single butterfly (output at stage 4
        MIN_get_routes(topology, ID, routes);
        for (int input = 0; input < N_input; input++)
            output_middle_stage[input] = (routes[input * n_stages + n_stagesHalf - 1] + rot_idx) % N_input;
        MIN_modify_conf(topology, output_middle_stage, ID_rot);
    }
    else {
        // complement middle stage + 1  of ID conf 
        bool* ID_modified = new bool[N_nodes * n_stages];
        
        for (int sw = 0; sw < N_nodes; sw++) {
            for (int stage = 0; stage < n_stages; stage++) {
                if (stage == n_stagesHalf) ID_modified[sw * n_stages + stage] = 1 - ID[sw * n_stages + stage];
                else ID_modified[sw * n_stages + stage] = ID[sw * n_stages + stage];
            }
        }
        for (int stage = 0; stage < n_stages; stage++) {
            for (int sw = 0; sw < N_nodes; sw++) {
                ID_rot[sw * n_stages + stage] = ID_modified[sw * n_stages + stage];
            }
        }
        MIN_get_routes(topology, ID_modified, routes);
        for (int input = 0; input < N_input; input++)
            output_middle_stage[input] = routes[input * n_stages + n_stagesHalf - 1];
        
        for (int input = 0; input < N_input; input++) {
            int item = routes[input * n_stages + n_stages - 1];
            for (int input2 = 0; input2 < N_input; input2++) {
                if (output_middle_stage[input2] == item) {
                    new_perm[input2] = (rot_idx + input) % N_input;
                    break;
                }
            }
        }
        MIN_modify_conf(topology, new_perm, ID_rot);

    }
}

void MIN_switch_identity(bool* ID_conf) {
    unsigned int N_input = 16;
    unsigned int n_stagesHalf = 4;//log(N_input)
    unsigned int n_stages = 2 * n_stagesHalf - 1;
    unsigned int N_nodes = N_input / 2;
    // init matrix with 0
    for (unsigned int i = 0; i < N_nodes; i++) {
        for (unsigned int j = 0; j < n_stages; j++) {
            ID_conf[i * n_stages + j] = 0;
        }
    }
    // first stage has all switches to 0
    // filling from the second to the last stage before the middle stage
    for (unsigned int k = 1; k < n_stagesHalf - 1; k++) {
        unsigned int items = pow(2,(n_stagesHalf - k - 2));
        for (unsigned int h = items; h < 2 * items; h++) {
            ID_conf[h * n_stages + k] = 1;
        }
        unsigned int copies = pow(2,(k - 1));
        while (copies >= 1) {
            unsigned int start = N_nodes / (2 * copies);
            for (unsigned int h = start; h < 2 * start; h++) {
                ID_conf[h * n_stages + k] = ID_conf[(2 * start - h-1) * 7 + k];
            }
            copies = copies / 2;
        }
    }
    // filling the middle stage
    for (unsigned int h = 1; h < N_nodes / 2; h = h + 2) {
        ID_conf[h * n_stages + n_stagesHalf-1] = 1;
        ID_conf[(N_nodes - 1 - h) * n_stages + n_stagesHalf-1] = 1;
    }
    // copying second half of the network configuration
    for (unsigned int k = 1; k < n_stagesHalf; k++){
        for (unsigned int h = 0; h < N_nodes; h++) {
            ID_conf[h * n_stages + (k + n_stagesHalf - 1)] = ID_conf[h * n_stages + k];
        }
    }

}

void generate_rotation_configurations(int* topology, bool* out) {

    /*
        HOW TO CALL example: 
        int* TOP = new int[16 * 6];
        make_butterfly_butterfly_topology(TOP);

        bool* char_mat = new bool[16 * 8 * 7];
        generate_rotation_configurations(TOP, char_mat);
        print_bool_matrix(char_mat, 16 * 8, 7);
    */
    bool* ID = new bool[8 * 7];
    MIN_switch_identity(ID);
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 7; j++) {
            out[(0 * 8 * 7) + (i * 7) + j] = ID[(i * 7) + j];
        }
    }

    for (int rot_idx = 1; rot_idx < 16; rot_idx++) {
        bool* rot = new bool[8 * 7];
        ID_rotation_configuration(topology, ID, rot_idx, rot);
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 7; j++) {
                out[(rot_idx * 8 * 7) + (i * 7) + j] = rot[(i * 7) + j];
            }
        }
    }
}