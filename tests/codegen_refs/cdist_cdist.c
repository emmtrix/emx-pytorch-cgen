#include <stdint.h>
#include <sys/types.h>
#include <stdbool.h>
#include "ops_scalar_f32.h"

/*
* op: _cdist_forward (kind: cdist)
* inputs: [shape=(2, 3), size=6, shape=(4, 3), size=12]
* output: shape=(2, 4), size=8
* params: {'p': 2.0, 'compute_mode': None}
*/
void node1__cdist_forward_f32(const float x1[2][3], const float x2[4][3], float out[2][4]) {
    for (ssize_t i = 0; i < 2; ++i) {
        for (ssize_t j = 0; j < 4; ++j) {
            float acc = 0;
            for (ssize_t k = 0; k < 3; ++k) {
                float diff = ((float*)x1)[i * 3 + k * 1] - ((float*)x2)[j * 3 + k * 1];
                acc += diff * diff;
            }
            out[i][j] = ref_scalar_f32_sqrt(acc);
        }
    }
}

void ref_codegen_main_f32(const float input_0[2][3], const float input_1[4][3], float out[2][4]) {
    node1__cdist_forward_f32(input_0, input_1, out);
}
