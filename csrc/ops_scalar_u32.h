#ifndef REF_BACKEND_OPS_SCALAR_U32_H
#define REF_BACKEND_OPS_SCALAR_U32_H

#include <stdint.h>

#include "ops_scalar_f32.h"

static inline uint32_t ref_scalar_u32_from_f32(float value) {
    if (!isfinite(value)) {
        return 0;
    }
    if (value <= 0.0f) {
        return 0;
    }
    if (value >= (float)UINT32_MAX) {
        return UINT32_MAX;
    }
    return (uint32_t)value;
}

static inline uint32_t ref_scalar_u32_abs(uint32_t a) {
    return a;
}

static inline uint32_t ref_scalar_u32_absolute(uint32_t a) {
    return a;
}

static inline uint32_t ref_scalar_u32_add(uint32_t a, uint32_t b) {
    return a + b;
}

static inline uint32_t ref_scalar_u32_sub(uint32_t a, uint32_t b) {
    return a - b;
}

static inline uint32_t ref_scalar_u32_mul(uint32_t a, uint32_t b) {
    return a * b;
}

static inline uint32_t ref_scalar_u32_bitwise_and(uint32_t a, uint32_t b) {
    return a & b;
}

static inline uint32_t ref_scalar_u32_bitwise_or(uint32_t a, uint32_t b) {
    return a | b;
}

static inline uint32_t ref_scalar_u32_bitwise_xor(uint32_t a, uint32_t b) {
    return a ^ b;
}

static inline uint32_t ref_scalar_u32_bitwise_left_shift(uint32_t a, uint32_t b) {
    return a << b;
}

static inline uint32_t ref_scalar_u32_bitwise_right_shift(uint32_t a, uint32_t b) {
    return a >> b;
}

static inline uint32_t ref_scalar_u32_bitwise_not(uint32_t a) {
    return ~a;
}

static inline uint32_t ref_scalar_u32_div(uint32_t a, uint32_t b) {
    if (b == 0) {
        return 0;
    }
    return a / b;
}

static inline uint32_t ref_scalar_u32_maximum(uint32_t a, uint32_t b) {
    return a > b ? a : b;
}

static inline uint32_t ref_scalar_u32_minimum(uint32_t a, uint32_t b) {
    return a < b ? a : b;
}

static inline uint32_t ref_scalar_u32_le(uint32_t a, uint32_t b) {
    return a <= b ? (uint32_t)1 : (uint32_t)0;
}

static inline uint32_t ref_scalar_u32_lt(uint32_t a, uint32_t b) {
    return a < b ? (uint32_t)1 : (uint32_t)0;
}

static inline uint32_t ref_scalar_u32_ge(uint32_t a, uint32_t b) {
    return a >= b ? (uint32_t)1 : (uint32_t)0;
}

static inline uint32_t ref_scalar_u32_gt(uint32_t a, uint32_t b) {
    return a > b ? (uint32_t)1 : (uint32_t)0;
}

static inline uint32_t ref_scalar_u32_eq(uint32_t a, uint32_t b) {
    return a == b ? (uint32_t)1 : (uint32_t)0;
}

static inline uint32_t ref_scalar_u32_ne(uint32_t a, uint32_t b) {
    return a != b ? (uint32_t)1 : (uint32_t)0;
}

static inline uint32_t ref_scalar_u32_logical_or(uint32_t a, uint32_t b) {
    return (a != 0 || b != 0) ? (uint32_t)1 : (uint32_t)0;
}

static inline uint32_t ref_scalar_u32_logical_and(uint32_t a, uint32_t b) {
    return (a != 0 && b != 0) ? (uint32_t)1 : (uint32_t)0;
}

static inline uint32_t ref_scalar_u32_logical_xor(uint32_t a, uint32_t b) {
    return ((a != 0) != (b != 0)) ? (uint32_t)1 : (uint32_t)0;
}

static inline uint32_t ref_scalar_u32_logical_not(uint32_t a) {
    return a == 0 ? (uint32_t)1 : (uint32_t)0;
}

static inline uint32_t ref_scalar_u32_fmax(uint32_t a, uint32_t b) {
    return a > b ? a : b;
}

static inline uint32_t ref_scalar_u32_fmin(uint32_t a, uint32_t b) {
    return a < b ? a : b;
}

static inline uint32_t ref_scalar_u32_copysign(uint32_t a, uint32_t b) {
    (void)b;
    return a;
}

static inline uint32_t ref_scalar_u32_fmod(uint32_t a, uint32_t b) {
    if (b == 0) {
        return 0;
    }
    return a % b;
}

static inline uint32_t ref_scalar_u32_remainder(uint32_t a, uint32_t b) {
    if (b == 0) {
        return 0;
    }
    return a % b;
}

static inline uint32_t ref_scalar_u32_floor_divide(uint32_t a, uint32_t b) {
    if (b == 0) {
        return 0;
    }
    return a / b;
}

static inline uint32_t ref_scalar_u32_clamp_min(uint32_t a, uint32_t b) {
    return a > b ? a : b;
}

static inline uint32_t ref_scalar_u32_clamp_max(uint32_t a, uint32_t b) {
    return a < b ? a : b;
}

static inline uint32_t ref_scalar_u32_neg(uint32_t a) {
    return (uint32_t)(0 - a);
}

static inline uint32_t ref_scalar_u32_reciprocal(uint32_t a) {
    if (a == 0) {
        return 0;
    }
    return 1 / a;
}

static inline uint32_t ref_scalar_u32_relu(uint32_t a) {
    return a;
}

static inline uint32_t ref_scalar_u32_ceil(uint32_t a) {
    return a;
}

static inline uint32_t ref_scalar_u32_floor(uint32_t a) {
    return a;
}

static inline uint32_t ref_scalar_u32_round(uint32_t a) {
    return a;
}

#define REF_U32_UNARY_FROM_F32(name)                         \
    static inline uint32_t ref_scalar_u32_##name(uint32_t a) { \
        return ref_scalar_u32_from_f32(ref_scalar_f32_##name( \
            (float)a));                                      \
    }

#define REF_U32_BINARY_FROM_F32(name)                                 \
    static inline uint32_t ref_scalar_u32_##name(uint32_t a, uint32_t b) { \
        return ref_scalar_u32_from_f32(ref_scalar_f32_##name(         \
            (float)a, (float)b));                                    \
    }

REF_U32_UNARY_FROM_F32(acos)
REF_U32_UNARY_FROM_F32(arccos)
REF_U32_UNARY_FROM_F32(acosh)
REF_U32_UNARY_FROM_F32(angle)
REF_U32_UNARY_FROM_F32(asin)
REF_U32_UNARY_FROM_F32(arcsin)
REF_U32_UNARY_FROM_F32(asinh)
REF_U32_UNARY_FROM_F32(arcsinh)
REF_U32_UNARY_FROM_F32(atan)
REF_U32_UNARY_FROM_F32(arctan)
REF_U32_UNARY_FROM_F32(atanh)
REF_U32_UNARY_FROM_F32(cbrt)
REF_U32_UNARY_FROM_F32(cos)
REF_U32_UNARY_FROM_F32(cosh)
REF_U32_UNARY_FROM_F32(deg2rad)
REF_U32_UNARY_FROM_F32(digamma)
REF_U32_UNARY_FROM_F32(erf)
REF_U32_UNARY_FROM_F32(erfc)
REF_U32_UNARY_FROM_F32(erfinv)
REF_U32_UNARY_FROM_F32(exp)
REF_U32_UNARY_FROM_F32(exp2)
REF_U32_UNARY_FROM_F32(expm1)
REF_U32_UNARY_FROM_F32(i0)
REF_U32_UNARY_FROM_F32(lgamma)
REF_U32_UNARY_FROM_F32(log)
REF_U32_UNARY_FROM_F32(log10)
REF_U32_UNARY_FROM_F32(log1p)
REF_U32_UNARY_FROM_F32(log2)
REF_U32_UNARY_FROM_F32(isfinite)
REF_U32_UNARY_FROM_F32(isnan)
REF_U32_UNARY_FROM_F32(logit)
REF_U32_UNARY_FROM_F32(log_sigmoid)
REF_U32_UNARY_FROM_F32(gelu)
REF_U32_UNARY_FROM_F32(elu)
REF_U32_UNARY_FROM_F32(leaky_relu)
REF_U32_UNARY_FROM_F32(softplus)
REF_U32_UNARY_FROM_F32(isinf)
REF_U32_UNARY_FROM_F32(isneginf)
REF_U32_UNARY_FROM_F32(isposinf)
REF_U32_UNARY_FROM_F32(nan_to_num)
REF_U32_UNARY_FROM_F32(rad2deg)
REF_U32_UNARY_FROM_F32(rsqrt)
REF_U32_UNARY_FROM_F32(sigmoid)
REF_U32_UNARY_FROM_F32(selu)
REF_U32_UNARY_FROM_F32(relu6)
REF_U32_UNARY_FROM_F32(hardsigmoid)
REF_U32_UNARY_FROM_F32(silu)
REF_U32_UNARY_FROM_F32(mish)
REF_U32_UNARY_FROM_F32(hardswish)
REF_U32_UNARY_FROM_F32(sin)
REF_U32_UNARY_FROM_F32(sinc)
REF_U32_UNARY_FROM_F32(sinh)
REF_U32_UNARY_FROM_F32(sqrt)
REF_U32_UNARY_FROM_F32(tan)
REF_U32_UNARY_FROM_F32(tanh)

REF_U32_BINARY_FROM_F32(atan2)
REF_U32_BINARY_FROM_F32(heaviside)
REF_U32_BINARY_FROM_F32(hypot)
REF_U32_BINARY_FROM_F32(ldexp)
REF_U32_BINARY_FROM_F32(logaddexp)
REF_U32_BINARY_FROM_F32(logaddexp2)
REF_U32_BINARY_FROM_F32(nextafter)
REF_U32_BINARY_FROM_F32(pow)
REF_U32_BINARY_FROM_F32(xlogy)

#undef REF_U32_UNARY_FROM_F32
#undef REF_U32_BINARY_FROM_F32

#endif
