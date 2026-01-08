#ifndef REF_BACKEND_OPS_SCALAR_U64_H
#define REF_BACKEND_OPS_SCALAR_U64_H

#include <stdint.h>

#include "ops_scalar_f32.h"

static inline uint64_t ref_scalar_u64_from_f32(float value) {
    if (!isfinite(value)) {
        return 0;
    }
    if (value <= 0.0f) {
        return 0;
    }
    if (value >= (float)UINT64_MAX) {
        return UINT64_MAX;
    }
    return (uint64_t)value;
}

static inline uint64_t ref_scalar_u64_abs(uint64_t a) {
    return a;
}

static inline uint64_t ref_scalar_u64_absolute(uint64_t a) {
    return a;
}

static inline uint64_t ref_scalar_u64_add(uint64_t a, uint64_t b) {
    return a + b;
}

static inline uint64_t ref_scalar_u64_sub(uint64_t a, uint64_t b) {
    return a - b;
}

static inline uint64_t ref_scalar_u64_mul(uint64_t a, uint64_t b) {
    return a * b;
}

static inline uint64_t ref_scalar_u64_bitwise_and(uint64_t a, uint64_t b) {
    return a & b;
}

static inline uint64_t ref_scalar_u64_bitwise_or(uint64_t a, uint64_t b) {
    return a | b;
}

static inline uint64_t ref_scalar_u64_bitwise_xor(uint64_t a, uint64_t b) {
    return a ^ b;
}

static inline uint64_t ref_scalar_u64_bitwise_left_shift(uint64_t a, uint64_t b) {
    return a << b;
}

static inline uint64_t ref_scalar_u64_bitwise_right_shift(uint64_t a, uint64_t b) {
    return a >> b;
}

static inline uint64_t ref_scalar_u64_bitwise_not(uint64_t a) {
    return ~a;
}

static inline uint64_t ref_scalar_u64_div(uint64_t a, uint64_t b) {
    if (b == 0) {
        return 0;
    }
    return a / b;
}

static inline uint64_t ref_scalar_u64_maximum(uint64_t a, uint64_t b) {
    return a > b ? a : b;
}

static inline uint64_t ref_scalar_u64_minimum(uint64_t a, uint64_t b) {
    return a < b ? a : b;
}

static inline uint64_t ref_scalar_u64_le(uint64_t a, uint64_t b) {
    return a <= b ? (uint64_t)1 : (uint64_t)0;
}

static inline uint64_t ref_scalar_u64_lt(uint64_t a, uint64_t b) {
    return a < b ? (uint64_t)1 : (uint64_t)0;
}

static inline uint64_t ref_scalar_u64_ge(uint64_t a, uint64_t b) {
    return a >= b ? (uint64_t)1 : (uint64_t)0;
}

static inline uint64_t ref_scalar_u64_gt(uint64_t a, uint64_t b) {
    return a > b ? (uint64_t)1 : (uint64_t)0;
}

static inline uint64_t ref_scalar_u64_eq(uint64_t a, uint64_t b) {
    return a == b ? (uint64_t)1 : (uint64_t)0;
}

static inline uint64_t ref_scalar_u64_ne(uint64_t a, uint64_t b) {
    return a != b ? (uint64_t)1 : (uint64_t)0;
}

static inline uint64_t ref_scalar_u64_logical_or(uint64_t a, uint64_t b) {
    return (a != 0 || b != 0) ? (uint64_t)1 : (uint64_t)0;
}

static inline uint64_t ref_scalar_u64_logical_and(uint64_t a, uint64_t b) {
    return (a != 0 && b != 0) ? (uint64_t)1 : (uint64_t)0;
}

static inline uint64_t ref_scalar_u64_logical_xor(uint64_t a, uint64_t b) {
    return ((a != 0) != (b != 0)) ? (uint64_t)1 : (uint64_t)0;
}

static inline uint64_t ref_scalar_u64_logical_not(uint64_t a) {
    return a == 0 ? (uint64_t)1 : (uint64_t)0;
}

static inline uint64_t ref_scalar_u64_fmax(uint64_t a, uint64_t b) {
    return a > b ? a : b;
}

static inline uint64_t ref_scalar_u64_fmin(uint64_t a, uint64_t b) {
    return a < b ? a : b;
}

static inline uint64_t ref_scalar_u64_copysign(uint64_t a, uint64_t b) {
    (void)b;
    return a;
}

static inline uint64_t ref_scalar_u64_fmod(uint64_t a, uint64_t b) {
    if (b == 0) {
        return 0;
    }
    return a % b;
}

static inline uint64_t ref_scalar_u64_remainder(uint64_t a, uint64_t b) {
    if (b == 0) {
        return 0;
    }
    return a % b;
}

static inline uint64_t ref_scalar_u64_floor_divide(uint64_t a, uint64_t b) {
    if (b == 0) {
        return 0;
    }
    return a / b;
}

static inline uint64_t ref_scalar_u64_clamp_min(uint64_t a, uint64_t b) {
    return a > b ? a : b;
}

static inline uint64_t ref_scalar_u64_clamp_max(uint64_t a, uint64_t b) {
    return a < b ? a : b;
}

static inline uint64_t ref_scalar_u64_neg(uint64_t a) {
    return (uint64_t)(0 - a);
}

static inline uint64_t ref_scalar_u64_reciprocal(uint64_t a) {
    if (a == 0) {
        return 0;
    }
    return 1 / a;
}

static inline uint64_t ref_scalar_u64_relu(uint64_t a) {
    return a;
}

static inline uint64_t ref_scalar_u64_ceil(uint64_t a) {
    return a;
}

static inline uint64_t ref_scalar_u64_floor(uint64_t a) {
    return a;
}

static inline uint64_t ref_scalar_u64_round(uint64_t a) {
    return a;
}

#define REF_U64_UNARY_FROM_F32(name)                         \
    static inline uint64_t ref_scalar_u64_##name(uint64_t a) { \
        return ref_scalar_u64_from_f32(ref_scalar_f32_##name( \
            (float)a));                                      \
    }

#define REF_U64_BINARY_FROM_F32(name)                                 \
    static inline uint64_t ref_scalar_u64_##name(uint64_t a, uint64_t b) { \
        return ref_scalar_u64_from_f32(ref_scalar_f32_##name(         \
            (float)a, (float)b));                                    \
    }

REF_U64_UNARY_FROM_F32(acos)
REF_U64_UNARY_FROM_F32(arccos)
REF_U64_UNARY_FROM_F32(acosh)
REF_U64_UNARY_FROM_F32(angle)
REF_U64_UNARY_FROM_F32(asin)
REF_U64_UNARY_FROM_F32(arcsin)
REF_U64_UNARY_FROM_F32(asinh)
REF_U64_UNARY_FROM_F32(arcsinh)
REF_U64_UNARY_FROM_F32(atan)
REF_U64_UNARY_FROM_F32(arctan)
REF_U64_UNARY_FROM_F32(atanh)
REF_U64_UNARY_FROM_F32(cbrt)
REF_U64_UNARY_FROM_F32(cos)
REF_U64_UNARY_FROM_F32(cosh)
REF_U64_UNARY_FROM_F32(deg2rad)
REF_U64_UNARY_FROM_F32(digamma)
REF_U64_UNARY_FROM_F32(erf)
REF_U64_UNARY_FROM_F32(erfc)
REF_U64_UNARY_FROM_F32(erfinv)
REF_U64_UNARY_FROM_F32(exp)
REF_U64_UNARY_FROM_F32(exp2)
REF_U64_UNARY_FROM_F32(expm1)
REF_U64_UNARY_FROM_F32(i0)
REF_U64_UNARY_FROM_F32(lgamma)
REF_U64_UNARY_FROM_F32(log)
REF_U64_UNARY_FROM_F32(log10)
REF_U64_UNARY_FROM_F32(log1p)
REF_U64_UNARY_FROM_F32(log2)
REF_U64_UNARY_FROM_F32(isfinite)
REF_U64_UNARY_FROM_F32(isnan)
REF_U64_UNARY_FROM_F32(logit)
REF_U64_UNARY_FROM_F32(log_sigmoid)
REF_U64_UNARY_FROM_F32(gelu)
REF_U64_UNARY_FROM_F32(elu)
REF_U64_UNARY_FROM_F32(leaky_relu)
REF_U64_UNARY_FROM_F32(softplus)
REF_U64_UNARY_FROM_F32(isinf)
REF_U64_UNARY_FROM_F32(isneginf)
REF_U64_UNARY_FROM_F32(isposinf)
REF_U64_UNARY_FROM_F32(nan_to_num)
REF_U64_UNARY_FROM_F32(rad2deg)
REF_U64_UNARY_FROM_F32(rsqrt)
REF_U64_UNARY_FROM_F32(sigmoid)
REF_U64_UNARY_FROM_F32(selu)
REF_U64_UNARY_FROM_F32(relu6)
REF_U64_UNARY_FROM_F32(hardsigmoid)
REF_U64_UNARY_FROM_F32(silu)
REF_U64_UNARY_FROM_F32(mish)
REF_U64_UNARY_FROM_F32(hardswish)
REF_U64_UNARY_FROM_F32(sin)
REF_U64_UNARY_FROM_F32(sinc)
REF_U64_UNARY_FROM_F32(sinh)
REF_U64_UNARY_FROM_F32(sqrt)
REF_U64_UNARY_FROM_F32(tan)
REF_U64_UNARY_FROM_F32(tanh)

REF_U64_BINARY_FROM_F32(atan2)
REF_U64_BINARY_FROM_F32(heaviside)
REF_U64_BINARY_FROM_F32(hypot)
REF_U64_BINARY_FROM_F32(ldexp)
REF_U64_BINARY_FROM_F32(logaddexp)
REF_U64_BINARY_FROM_F32(logaddexp2)
REF_U64_BINARY_FROM_F32(nextafter)
REF_U64_BINARY_FROM_F32(pow)
REF_U64_BINARY_FROM_F32(xlogy)

#undef REF_U64_UNARY_FROM_F32
#undef REF_U64_BINARY_FROM_F32

#endif
