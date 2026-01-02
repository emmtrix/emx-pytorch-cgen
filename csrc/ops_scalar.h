#ifndef REF_BACKEND_OPS_SCALAR_H
#define REF_BACKEND_OPS_SCALAR_H

#include <math.h>

static inline float ref_scalar_abs(float a) {
    return fabsf(a);
}

static inline float ref_scalar_add(float a, float b) {
    return a + b;
}

static inline float ref_scalar_sub(float a, float b) {
    return a - b;
}

static inline float ref_scalar_mul(float a, float b) {
    return a * b;
}

static inline float ref_scalar_div(float a, float b) {
    return a / b;
}

static inline float ref_scalar_maximum(float a, float b) {
    return fmaxf(a, b);
}

static inline float ref_scalar_minimum(float a, float b) {
    return fminf(a, b);
}

static inline float ref_scalar_neg(float a) {
    return -a;
}

static inline float ref_scalar_reciprocal(float a) {
    return 1.0f / a;
}

static inline float ref_scalar_relu(float a) {
    return a > 0.0f ? a : 0.0f;
}

static inline float ref_scalar_ceil(float a) {
    return ceilf(a);
}

static inline float ref_scalar_floor(float a) {
    return floorf(a);
}

static inline float ref_scalar_sin(float a) {
    return sinf(a);
}

static inline float ref_scalar_cos(float a) {
    return cosf(a);
}

static inline float ref_scalar_sqrt(float a) {
    return sqrtf(a);
}

static inline float ref_scalar_exp(float a) {
    return expf(a);
}

static inline float ref_scalar_tanh(float a) {
    return tanhf(a);
}

static inline float ref_scalar_log(float a) {
    return logf(a);
}

static inline float ref_scalar_acos(float a) {
    return acosf(a);
}

static inline float ref_scalar_acosh(float a) {
    return acoshf(a);
}

static inline float ref_scalar_asin(float a) {
    return asinf(a);
}

static inline float ref_scalar_asinh(float a) {
    return asinhf(a);
}

static inline float ref_scalar_atan(float a) {
    return atanf(a);
}

static inline float ref_scalar_atanh(float a) {
    return atanhf(a);
}

static inline float ref_scalar_cosh(float a) {
    return coshf(a);
}

static inline float ref_scalar_sinh(float a) {
    return sinhf(a);
}

static inline float ref_scalar_tan(float a) {
    return tanf(a);
}

static inline float ref_scalar_erf(float a) {
    return erff(a);
}

static inline float ref_scalar_erfc(float a) {
    return erfcf(a);
}

static inline float ref_scalar_expm1(float a) {
    return expm1f(a);
}

static inline float ref_scalar_log1p(float a) {
    return log1pf(a);
}

static inline float ref_scalar_log2(float a) {
    return log2f(a);
}

static inline float ref_scalar_log10(float a) {
    return log10f(a);
}

static inline float ref_scalar_rsqrt(float a) {
    return 1.0f / sqrtf(a);
}

static inline float ref_scalar_sigmoid(float a) {
    return 1.0f / (1.0f + expf(-a));
}

static inline float ref_scalar_sign(float a) {
    if (isnan(a)) {
        return a;
    }
    if (a > 0.0f) {
        return 1.0f;
    }
    if (a < 0.0f) {
        return -1.0f;
    }
    return 0.0f;
}

static inline float ref_scalar_round(float a) {
    return roundf(a);
}

static inline float ref_scalar_trunc(float a) {
    return truncf(a);
}

#endif
