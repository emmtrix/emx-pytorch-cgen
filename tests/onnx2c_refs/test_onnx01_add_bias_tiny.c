#include <stdint.h>
#include <stdbool.h>
#include "ops_scalar_f32.h"

static const float weight_initializers_onnx_initializer_0[4] = {
    0.10000000149011612f, -0.20000000298023224f, 0.30000001192092896f, 0.0f
};

void node1_add_f32(const float a[1][4], const float b[4], float out[1][4]) {
    for (int64_t i0 = 0; i0 < 1; ++i0) {
        for (int64_t i1 = 0; i1 < 4; ++i1) {
            out[i0][i1] = ref_scalar_f32_add(a[0][i1], b[i1]);
        }
    }
}

void ref_codegen_main_f32(const float input_0[1][4], const float input_1[4], float out[1][4]) {
    node1_add_f32(input_0, input_1, out);
}

void model_run(const float* in0, float* out0) {
    ref_codegen_main_f32(in0, weight_initializers_onnx_initializer_0, out0);
}
