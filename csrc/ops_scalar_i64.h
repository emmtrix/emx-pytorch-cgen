#ifndef REF_BACKEND_OPS_SCALAR_I64_H
#define REF_BACKEND_OPS_SCALAR_I64_H

#include <limits.h>
#include <stdint.h>

#include "ops_scalar_f32.h"

static inline int64_t ref_scalar_i64_from_f32(float value) {
    if (!isfinite(value)) {
        return INT64_MIN;
    }
    if (value > (float)INT64_MAX) {
        return INT64_MAX;
    }
    if (value < (float)INT64_MIN) {
        return INT64_MIN;
    }
    return (int64_t)value;
}

static inline int64_t ref_scalar_i64_abs(int64_t a) {
    if (a == INT64_MIN) {
        return INT64_MIN;
    }
    return a < 0 ? -a : a;
}

static inline int64_t ref_scalar_i64_absolute(int64_t a) {
    return ref_scalar_i64_abs(a);
}

static inline int64_t ref_scalar_i64_add(int64_t a, int64_t b) {
    return a + b;
}

static inline int64_t ref_scalar_i64_sub(int64_t a, int64_t b) {
    return a - b;
}

static inline int64_t ref_scalar_i64_mul(int64_t a, int64_t b) {
    return a * b;
}

static inline int64_t ref_scalar_i64_bitwise_and(int64_t a, int64_t b) {
    return a & b;
}

static inline int64_t ref_scalar_i64_bitwise_or(int64_t a, int64_t b) {
    return a | b;
}

static inline int64_t ref_scalar_i64_bitwise_xor(int64_t a, int64_t b) {
    return a ^ b;
}

static inline int64_t ref_scalar_i64_bitwise_left_shift(int64_t a, int64_t b) {
    return a << b;
}

static inline int64_t ref_scalar_i64_bitwise_right_shift(int64_t a, int64_t b) {
    return a >> b;
}

static inline int64_t ref_scalar_i64_bitwise_not(int64_t a) {
    return ~a;
}

static inline int64_t ref_scalar_i64_div(int64_t a, int64_t b) {
    if (b == 0) {
        return 0;
    }
    return a / b;
}

static inline int64_t ref_scalar_i64_maximum(int64_t a, int64_t b) {
    return a > b ? a : b;
}

static inline int64_t ref_scalar_i64_minimum(int64_t a, int64_t b) {
    return a < b ? a : b;
}

static inline int64_t ref_scalar_i64_le(int64_t a, int64_t b) {
    return a <= b ? 1 : 0;
}

static inline int64_t ref_scalar_i64_lt(int64_t a, int64_t b) {
    return a < b ? 1 : 0;
}

static inline int64_t ref_scalar_i64_ge(int64_t a, int64_t b) {
    return a >= b ? 1 : 0;
}

static inline int64_t ref_scalar_i64_gt(int64_t a, int64_t b) {
    return a > b ? 1 : 0;
}

static inline int64_t ref_scalar_i64_eq(int64_t a, int64_t b) {
    return a == b ? 1 : 0;
}

static inline int64_t ref_scalar_i64_ne(int64_t a, int64_t b) {
    return a != b ? 1 : 0;
}

static inline int64_t ref_scalar_i64_logical_or(int64_t a, int64_t b) {
    return (a != 0 || b != 0) ? 1 : 0;
}

static inline int64_t ref_scalar_i64_logical_and(int64_t a, int64_t b) {
    return (a != 0 && b != 0) ? 1 : 0;
}

static inline int64_t ref_scalar_i64_logical_xor(int64_t a, int64_t b) {
    return ((a != 0) != (b != 0)) ? 1 : 0;
}

static inline int64_t ref_scalar_i64_logical_not(int64_t a) {
    return a == 0 ? 1 : 0;
}

static inline int64_t ref_scalar_i64_fmax(int64_t a, int64_t b) {
    return a > b ? a : b;
}

static inline int64_t ref_scalar_i64_fmin(int64_t a, int64_t b) {
    return a < b ? a : b;
}

static inline int64_t ref_scalar_i64_copysign(int64_t a, int64_t b) {
    int64_t magnitude = ref_scalar_i64_abs(a);
    return b < 0 ? -magnitude : magnitude;
}

static inline int64_t ref_scalar_i64_fmod(int64_t a, int64_t b) {
    if (b == 0) {
        return 0;
    }
    return a % b;
}

static inline int64_t ref_scalar_i64_remainder(int64_t a, int64_t b) {
    if (b == 0) {
        return 0;
    }
    int64_t mod = a % b;
    if (mod == 0) {
        return mod;
    }
    if ((mod < 0) != (b < 0)) {
        mod += b;
    }
    return mod;
}

static inline int64_t ref_scalar_i64_floor_divide(int64_t a, int64_t b) {
    if (b == 0) {
        return 0;
    }
    int64_t quo = a / b;
    int64_t rem = a % b;
    if (rem != 0 && ((rem < 0) != (b < 0))) {
        quo -= 1;
    }
    return quo;
}

static inline int64_t ref_scalar_i64_clamp_min(int64_t a, int64_t b) {
    return a > b ? a : b;
}

static inline int64_t ref_scalar_i64_clamp_max(int64_t a, int64_t b) {
    return a < b ? a : b;
}

static inline int64_t ref_scalar_i64_neg(int64_t a) {
    if (a == INT64_MIN) {
        return INT64_MIN;
    }
    return -a;
}

static inline int64_t ref_scalar_i64_reciprocal(int64_t a) {
    if (a == 0) {
        return 0;
    }
    return 1 / a;
}

static inline int64_t ref_scalar_i64_relu(int64_t a) {
    return a > 0 ? a : 0;
}

static inline int64_t ref_scalar_i64_ceil(int64_t a) {
    return a;
}

static inline int64_t ref_scalar_i64_floor(int64_t a) {
    return a;
}

static inline int64_t ref_scalar_i64_round(int64_t a) {
    return a;
}

static inline int64_t ref_scalar_i64_trunc(int64_t a) {
    return a;
}

static inline int64_t ref_scalar_i64_frac(int64_t a) {
    (void)a;
    return 0;
}

static inline int64_t ref_scalar_i64_sign(int64_t a) {
    if (a > 0) {
        return 1;
    }
    if (a < 0) {
        return -1;
    }
    return 0;
}

static inline int64_t ref_scalar_i64_conj(int64_t a) {
    return a;
}

static inline int64_t ref_scalar_i64_conj_physical(int64_t a) {
    return a;
}

static inline int64_t ref_scalar_i64_positive(int64_t a) {
    return a;
}

static inline int64_t ref_scalar_i64_real(int64_t a) {
    return a;
}

static inline int64_t ref_scalar_i64_sgn(int64_t a) {
    if (a > 0) {
        return 1;
    }
    if (a < 0) {
        return -1;
    }
    return 0;
}

static inline int64_t ref_scalar_i64_square(int64_t a) {
    return a * a;
}

#define REF_I64_UNARY_FROM_F32(name)                          \
    static inline int64_t ref_scalar_i64_##name(int64_t a) {   \
        return ref_scalar_i64_from_f32(ref_scalar_f32_##name(  \
            (float)a));                                       \
    }

#define REF_I64_BINARY_FROM_F32(name)                                 \
    static inline int64_t ref_scalar_i64_##name(int64_t a, int64_t b) { \
        return ref_scalar_i64_from_f32(ref_scalar_f32_##name(         \
            (float)a, (float)b));                                     \
    }

REF_I64_UNARY_FROM_F32(acos)
REF_I64_UNARY_FROM_F32(arccos)
REF_I64_UNARY_FROM_F32(acosh)
REF_I64_UNARY_FROM_F32(angle)
REF_I64_UNARY_FROM_F32(asin)
REF_I64_UNARY_FROM_F32(arcsin)
REF_I64_UNARY_FROM_F32(asinh)
REF_I64_UNARY_FROM_F32(arcsinh)
REF_I64_UNARY_FROM_F32(atan)
REF_I64_UNARY_FROM_F32(arctan)
REF_I64_UNARY_FROM_F32(atanh)
REF_I64_UNARY_FROM_F32(cbrt)
REF_I64_UNARY_FROM_F32(cos)
REF_I64_UNARY_FROM_F32(cosh)
REF_I64_UNARY_FROM_F32(deg2rad)
REF_I64_UNARY_FROM_F32(digamma)
REF_I64_UNARY_FROM_F32(erf)
REF_I64_UNARY_FROM_F32(erfc)
REF_I64_UNARY_FROM_F32(erfinv)
REF_I64_UNARY_FROM_F32(exp)
REF_I64_UNARY_FROM_F32(exp2)
REF_I64_UNARY_FROM_F32(expm1)
REF_I64_UNARY_FROM_F32(i0)
REF_I64_UNARY_FROM_F32(lgamma)
REF_I64_UNARY_FROM_F32(log)
REF_I64_UNARY_FROM_F32(log10)
REF_I64_UNARY_FROM_F32(log1p)
REF_I64_UNARY_FROM_F32(log2)
REF_I64_UNARY_FROM_F32(isfinite)
REF_I64_UNARY_FROM_F32(isnan)
REF_I64_UNARY_FROM_F32(logit)
REF_I64_UNARY_FROM_F32(log_sigmoid)
REF_I64_UNARY_FROM_F32(gelu)
REF_I64_UNARY_FROM_F32(elu)
REF_I64_UNARY_FROM_F32(leaky_relu)
REF_I64_UNARY_FROM_F32(softplus)
REF_I64_UNARY_FROM_F32(isinf)
REF_I64_UNARY_FROM_F32(isneginf)
REF_I64_UNARY_FROM_F32(isposinf)
REF_I64_UNARY_FROM_F32(nan_to_num)
REF_I64_UNARY_FROM_F32(rad2deg)
REF_I64_UNARY_FROM_F32(rsqrt)
REF_I64_UNARY_FROM_F32(sigmoid)
REF_I64_UNARY_FROM_F32(selu)
REF_I64_UNARY_FROM_F32(relu6)
REF_I64_UNARY_FROM_F32(hardsigmoid)
REF_I64_UNARY_FROM_F32(silu)
REF_I64_UNARY_FROM_F32(mish)
REF_I64_UNARY_FROM_F32(hardswish)
REF_I64_UNARY_FROM_F32(sin)
REF_I64_UNARY_FROM_F32(sinc)
REF_I64_UNARY_FROM_F32(sinh)
REF_I64_UNARY_FROM_F32(sqrt)
REF_I64_UNARY_FROM_F32(tan)
REF_I64_UNARY_FROM_F32(tanh)

REF_I64_BINARY_FROM_F32(atan2)
REF_I64_BINARY_FROM_F32(heaviside)
REF_I64_BINARY_FROM_F32(hypot)
REF_I64_BINARY_FROM_F32(ldexp)
REF_I64_BINARY_FROM_F32(logaddexp)
REF_I64_BINARY_FROM_F32(logaddexp2)
REF_I64_BINARY_FROM_F32(nextafter)
REF_I64_BINARY_FROM_F32(pow)
REF_I64_BINARY_FROM_F32(xlogy)

#undef REF_I64_UNARY_FROM_F32
#undef REF_I64_BINARY_FROM_F32

#endif
