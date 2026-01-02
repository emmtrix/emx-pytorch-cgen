#include "ops_binary.h"
#include "ops_scalar_f32.h"

int ref_run_add(const RefOpCall *call, char *err_msg, size_t err_cap) {
    return ref_run_binary_f32(call, err_msg, err_cap, ref_scalar_f32_add, "add");
}
