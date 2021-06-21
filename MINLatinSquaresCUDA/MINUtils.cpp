#include <algorithm>
#include <iterator>

#include "MINUtils.h"


void generateCharacteristicMatrices(unsigned int N, bool* out) {
	// auto N = 16;
	bool row[7];
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
}


void generateButterflyButterflyTopology(int* topology) {
    // topology(p,st) = arrival port at stage st+1 of outgoing link at port p of stage st
    for (int st = 0; st < 7; st++) {
        for (int sw = 0; sw < 8; sw++) {
            // selecting upper and lower port of switch sw at stage st
            port_up = sw * 2;
            port_down = sw * 2 + 1;
            // straight edge goes to switch sw at stage st+1
            straight_sw = sw;
            // cross edge goes to switch sw' at stage st+1 where sw' = sw, but the bit (st mod 4) is complemented
            bit_st = ((sw & (1 << (st%4 - 1))) >> (st%4 - 1));
            int mask = 1 << st%4;
            cross_sw = ((sw & ~mask) | (!bit_st << st%4));
            // find the arrival ports
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
                cross_port_depart = port_down;
                cross_port_arrival = cross_sw * 2 + 1; // lower port of arrival sw
            }
            topology(straight_port, st) = straight_port;
            topology(cross_port_depart, st) = cross_port_arrival;
        }
    }
}