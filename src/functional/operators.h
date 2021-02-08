#pragma once

#include "common/types.h"
#include <cmath>

namespace marian {
namespace functional {

// General template, will be used for any type without specializations
// and will fail at runtime with an abort message. Note that the
// general template functions don't have named parameters on purpose,
// because clang will warn about unused parameters during compilation.

template <typename T>
struct Ops {
  static HOST_DEVICE_INLINE T tanh(const T&) { ABORT("Unknown type"); }
  static HOST_DEVICE_INLINE T sin(const T&)  { ABORT("Unknown type"); }
  static HOST_DEVICE_INLINE T cos(const T&)  { ABORT("Unknown type"); }
  static HOST_DEVICE_INLINE T tan(const T&)  { ABORT("Unknown type"); }
  static HOST_DEVICE_INLINE T log(const T&)  { ABORT("Unknown type"); }
  static HOST_DEVICE_INLINE T exp(const T&)  { ABORT("Unknown type"); }
  static HOST_DEVICE_INLINE T abs(const T&)  { ABORT("Unknown type"); }
  static HOST_DEVICE_INLINE T sqr(const T&)  { ABORT("Unknown type"); }
  static HOST_DEVICE_INLINE T sqrt(const T&) { ABORT("Unknown type"); }
  static HOST_DEVICE_INLINE T neg(const T&)  { ABORT("Unknown type"); }
  static HOST_DEVICE_INLINE T sgn(const T&)  { ABORT("Unknown type"); }

  static HOST_DEVICE_INLINE T round(const T&)  { ABORT("Unknown type"); }
  static HOST_DEVICE_INLINE T floor(const T&)  { ABORT("Unknown type"); }
  static HOST_DEVICE_INLINE T ceil(const T&)   { ABORT("Unknown type"); }

  static HOST_DEVICE_INLINE T add(const T&, const T&)  { ABORT("Unknown type"); }
  static HOST_DEVICE_INLINE T sub(const T&, const T&)  { ABORT("Unknown type"); }
  static HOST_DEVICE_INLINE T mul(const T&, const T&)  { ABORT("Unknown type"); }
  static HOST_DEVICE_INLINE T div(const T&, const T&)  { ABORT("Unknown type"); }

  static HOST_DEVICE_INLINE T max(const T&, const T&)  { ABORT("Unknown type"); }
  static HOST_DEVICE_INLINE T min(const T&, const T&)  { ABORT("Unknown type"); }
  static HOST_DEVICE_INLINE T pow(const T&, const T&)  { ABORT("Unknown type"); }

  static HOST_DEVICE_INLINE T negate(const T&)  { ABORT("Unknown type"); }
  static HOST_DEVICE_INLINE T eq(const T&, const T&)   { ABORT("Unknown type"); }
  static HOST_DEVICE_INLINE T neq(const T&, const T&)  { ABORT("Unknown type"); }
  static HOST_DEVICE_INLINE T gt(const T&, const T&)   { ABORT("Unknown type"); }
  static HOST_DEVICE_INLINE T lt(const T&, const T&)   { ABORT("Unknown type"); }
  static HOST_DEVICE_INLINE T geq(const T&, const T&)  { ABORT("Unknown type"); }
  static HOST_DEVICE_INLINE T leq(const T&, const T&)  { ABORT("Unknown type"); }
  static HOST_DEVICE_INLINE T _and(const T&, const T&) { ABORT("Unknown type"); } // 'and' is used by gcc
  static HOST_DEVICE_INLINE T _or(const T&, const T&)  { ABORT("Unknown type"); } // 'or' is used by gcc

  // Neural Networks specific functions
  static HOST_DEVICE_INLINE T sigmoid(const T&)               { ABORT("Unknown type"); }
  static HOST_DEVICE_INLINE T logaddexp(const T&, const T&) { ABORT("Unknown type"); }
  static HOST_DEVICE_INLINE T clip(const T&, const T&)      { ABORT("Unknown type"); }
  // derivative of Clip, cut-off function
  static HOST_DEVICE_INLINE T bump(const T&, const T&)      { ABORT("Unknown type"); }
  static HOST_DEVICE_INLINE T relu(const T&)                  { ABORT("Unknown type"); }
  static HOST_DEVICE_INLINE T reluBack(const T&)              { ABORT("Unknown type"); }
  static HOST_DEVICE_INLINE T prelu(const T&, const T&)     { ABORT("Unknown type"); }
  static HOST_DEVICE_INLINE T preluBack(const T&, const T&) { ABORT("Unknown type"); }

  static HOST_DEVICE_INLINE T if_then_else(const T&, const T&, const T&) { ABORT("Unknown type"); }

  static HOST_DEVICE_INLINE T sumReduce(const T&) { ABORT("Unknown type"); }
  static HOST_DEVICE_INLINE T maxReduce(const T&) { ABORT("Unknown type"); }
  static HOST_DEVICE_INLINE T minReduce(const T&) { ABORT("Unknown type"); }
};

// Specialization for float
template <>
struct Ops<float> {
  typedef float Single;

  static HOST_DEVICE_INLINE float tanh(const float& x) { return tanhf(x); }
  static HOST_DEVICE_INLINE float sin(const float& x)  { return sinf(x); }
  static HOST_DEVICE_INLINE float cos(const float& x)  { return cosf(x); }
  static HOST_DEVICE_INLINE float tan(const float& x)  { return tanf(x); }
  static HOST_DEVICE_INLINE float log(const float& x)  { return logf(x); }
  static HOST_DEVICE_INLINE float exp(const float& x)  { return expf(x); }
  static HOST_DEVICE_INLINE float abs(const float& x)  { return fabs(x); }
  static HOST_DEVICE_INLINE float sqr(const float& x)  { return x * x; }
  static HOST_DEVICE_INLINE float sqrt(const float& x) { return sqrtf(x); }
  static HOST_DEVICE_INLINE float neg(const float& x)  { return -x; }
  static HOST_DEVICE_INLINE float sgn(const float& x)  { return (float)((0 < x) - (x < 0)); }

  static HOST_DEVICE_INLINE float round(const float& x)  { return roundf(x); }
  static HOST_DEVICE_INLINE float floor(const float& x)  { return floorf(x); }
  static HOST_DEVICE_INLINE float ceil(const float& x)   { return ceilf(x); }

  static HOST_DEVICE_INLINE float add(const float& x, const float& y)  { return x + y; }
  static HOST_DEVICE_INLINE float sub(const float& x, const float& y)  { return x - y; }
  static HOST_DEVICE_INLINE float mul(const float& x, const float& y)  { return x * y; }
  static HOST_DEVICE_INLINE float div(const float& x, const float& y)  { return x / y; }

  static HOST_DEVICE_INLINE float max(const float& x, const float& y)  { return x < y ? y : x; }
  static HOST_DEVICE_INLINE float min(const float& x, const float& y)  { return x < y ? x : y; }
  static HOST_DEVICE_INLINE float pow(const float& x, const float& y)  { return powf(x, y); }


  static HOST_DEVICE_INLINE float negate(const float& x)  { return !(bool)x; }
  static HOST_DEVICE_INLINE float eq(const float& x, const float& y)   { return x == y; }
  static HOST_DEVICE_INLINE float neq(const float& x, const float& y)  { return x != y; }
  static HOST_DEVICE_INLINE float gt(const float& x, const float& y)   { return x > y; }
  static HOST_DEVICE_INLINE float lt(const float& x, const float& y)   { return x < y; }
  static HOST_DEVICE_INLINE float geq(const float& x, const float& y)  { return x >= y; }
  static HOST_DEVICE_INLINE float leq(const float& x, const float& y)  { return x <= y; }
  static HOST_DEVICE_INLINE float and_(const float& x, const float& y) { return x && y; } // 'and' is used by gcc
  static HOST_DEVICE_INLINE float or_(const float& x, const float& y)  { return x || y; } // 'or' is used by gcc

  // Neural Networks specific functions
  static HOST_DEVICE_INLINE float sigmoid(const float& x) {
    return x > 0 ? (1.f / (1.f + exp(-x))) : (exp(x) / (1.f + exp(x)));
  }

  static HOST_DEVICE_INLINE float logaddexp(const float& x, const float& y) {
    // Note: This may not be ideal for CUDA; cf. CNTK implementation
    return x < y ? (y + log1pf(exp(x - y))) : (x + log1pf(exp(y - x)));
  }

  static HOST_DEVICE_INLINE float clip(const float& x, const float& y)  { return abs(x) >= y ? sgn(x) * y : x; }
  // derivative of Clip, cut-off function
  static HOST_DEVICE_INLINE float bump(const float& x, const float& y)  { return abs(x) >= y ? 0.f : 1.f; }

  static HOST_DEVICE_INLINE float relu(const float& x)     { return x > 0.f ? x : 0.f; }
  static HOST_DEVICE_INLINE float reluBack(const float& x) { return x > 0.f ? 1.f : 0.f; }

  static HOST_DEVICE_INLINE float prelu(const float& x, const float& y)     { return x > 0.f ? x : x * y; }
  static HOST_DEVICE_INLINE float preluBack(const float& x, const float& y) { return x > 0.f ? 1.f : y; }

  static HOST_DEVICE_INLINE float if_then_else(const float& x, const float& y, const float& z) { return x ? y : z; }

  static HOST_DEVICE_INLINE float sumReduce(const float& x) { return x; }
  static HOST_DEVICE_INLINE float maxReduce(const float& x) { return x; }
  static HOST_DEVICE_INLINE float minReduce(const float& x) { return x; }

};

// Specialization for double
template <>
struct Ops<double> {
  typedef double Single;

  static HOST_DEVICE_INLINE double tanh(const double& x) { return std::tanh(x); }
  static HOST_DEVICE_INLINE double sin(const double& x)  { return std::sin(x); }
  static HOST_DEVICE_INLINE double cos(const double& x)  { return std::cos(x); }
  static HOST_DEVICE_INLINE double tan(const double& x)  { return std::tan(x); }
  static HOST_DEVICE_INLINE double log(const double& x)  { return std::log(x); }
  static HOST_DEVICE_INLINE double exp(const double& x)  { return std::exp(x); }
  static HOST_DEVICE_INLINE double abs(const double& x)  { return std::abs(x); }
  static HOST_DEVICE_INLINE double sqr(const double& x)  { return x * x; }
  static HOST_DEVICE_INLINE double sqrt(const double& x) { return std::sqrt(x); }
  static HOST_DEVICE_INLINE double neg(const double& x)  { return -x; }
  static HOST_DEVICE_INLINE double sgn(const double& x)  { return (0 < x) - (x < 0); }

  static HOST_DEVICE_INLINE double round(const double& x)  { return std::round(x); }
  static HOST_DEVICE_INLINE double floor(const double& x)  { return std::floor(x); }
  static HOST_DEVICE_INLINE double ceil(const double& x)   { return std::ceil(x); }

  static HOST_DEVICE_INLINE double add(const double& x, const double& y)  { return x + y; }
  static HOST_DEVICE_INLINE double sub(const double& x, const double& y)  { return x - y; }
  static HOST_DEVICE_INLINE double mul(const double& x, const double& y)  { return x * y; }
  static HOST_DEVICE_INLINE double div(const double& x, const double& y)  { return x / y; }

  static HOST_DEVICE_INLINE double max(const double& x, const double& y)  { return x < y ? y : x; }
  static HOST_DEVICE_INLINE double min(const double& x, const double& y)  { return x < y ? x : y; }
  static HOST_DEVICE_INLINE double pow(const double& x, const double& y)  { return std::pow(x, y); }


  static HOST_DEVICE_INLINE double negate(const double& x)  { return !(bool)x; }
  static HOST_DEVICE_INLINE double eq(const double& x, const double& y)   { return x == y; }
  static HOST_DEVICE_INLINE double neq(const double& x, const double& y)  { return x != y; }
  static HOST_DEVICE_INLINE double gt(const double& x, const double& y)   { return x > y; }
  static HOST_DEVICE_INLINE double lt(const double& x, const double& y)   { return x < y; }
  static HOST_DEVICE_INLINE double geq(const double& x, const double& y)  { return x >= y; }
  static HOST_DEVICE_INLINE double leq(const double& x, const double& y)  { return x <= y; }
  static HOST_DEVICE_INLINE double and_(const double& x, const double& y) { return x && y; } // 'and' is used by gcc
  static HOST_DEVICE_INLINE double or_(const double& x, const double& y)  { return x || y; } // 'or' is used by gcc

  // Neural Networks specific functions
  static HOST_DEVICE_INLINE double sigmoid(const double& x) {
    return x > 0 ? (1.f / (1.f + exp(-x))) : (exp(x) / (1.f + exp(x)));
  }

  static HOST_DEVICE_INLINE double logaddexp(const double& x, const double& y) {
    // Note: This may not be ideal for CUDA; cf. CNTK implementation
    return x < y ? (y + log1p(exp(x - y))) : (x + log1p(exp(y - x)));
  }

  static HOST_DEVICE_INLINE double clip(const double& x, const double& y)  { return abs(x) >= y ? sgn(x) * y : x; }
  // derivative of Clip, cut-off function
  static HOST_DEVICE_INLINE double bump(const double& x, const double& y)  { return abs(x) >= y ? 0.f : 1.f; }

  static HOST_DEVICE_INLINE double relu(const double& x)     { return x > 0.f ? x : 0.f; }
  static HOST_DEVICE_INLINE double reluBack(const double& x) { return x > 0.f ? 1.f : 0.f; }

  static HOST_DEVICE_INLINE double prelu(const double& x, const double& y)     { return x > 0.f ? x : x * y; }
  static HOST_DEVICE_INLINE double preluBack(const double& x, const double& y) { return x > 0.f ? 1.f : y; }

  static HOST_DEVICE_INLINE double if_then_else(const double& x, const double& y, const double& z) { return x ? y : z; }

  static HOST_DEVICE_INLINE double sumReduce(const double& x) { return x; }
  static HOST_DEVICE_INLINE double maxReduce(const double& x) { return x; }
  static HOST_DEVICE_INLINE double minReduce(const double& x) { return x; }

};

} // end namespace functional
} // end namespace marian

// stay invisible to NVCC as it seems to have problems with intrinsics;
// will still be compiled into the binary by cpu-side gcc/g++
// __CUDACC__ is defined when compiling with NVCC regardless of device type
// __CUDA_ARCH__ is defined when compiling device (GPU) code
#ifndef __CUDACC__

#include "3rd_party/sse_mathfun.h"

namespace marian {
namespace functional {

// Specialization for float32x8 (=__m128, CPU SSE intrisics)
template <>
struct Ops<float32x4> {
  typedef float Single;

  static inline float32x4 loop4(const std::function<float(const float&)>& f, const float32x4& x) {
    float32x4 out;
    for(int i = 0; i < 4; i++)
      ((float*)&out)[i] = f(((const float*)&x)[i]);
    return out;
  }

  static inline float32x4 loop4(const std::function<float(const float&, const float&)>& f, const float32x4& x, const float32x4& y) {
    float32x4 out;
    for(int i = 0; i < 4; i++)
      ((float*)&out)[i] = f(((const float*)&x)[i], ((const float*)&y)[i]);
    return out;
  }

  static inline float32x4 loop4(const std::function<float(const float&, const float&, const float&)>& f, const float32x4& x, const float32x4& y, const float32x4& z) {
    float32x4 out;
    for(int i = 0; i < 4; i++)
      ((float*)&out)[i] = f(((const float*)&x)[i], ((const float*)&y)[i], ((const float*)&z)[i]);
    return out;
  }

  // @TODO: why is this slow?
  static inline float32x4 tanh(const float32x4& x) {
    // ( e^x - e^-x )/( e^x + e^-x ) = (e^2x - 1) / (e^2x + 1)
    float32x4 e2x = exp(mul(2.f, x));
    return div(sub(e2x, 1.f), add(e2x, 1.f));
  }

  static inline float32x4 sin(const float32x4& x) { return sin_ps(x); }
  static inline float32x4 cos(const float32x4& x) { return cos_ps(x); }
  static inline float32x4 tan(const float32x4& x) { return div(sin(x), cos(x)); }
  static inline float32x4 log(const float32x4& x) { return log_ps(x); }
  static inline float32x4 exp(const float32x4& x) { return exp_ps(x); }

  // @TODO: get rid of loop4 with proper intrisics
  static inline float32x4 abs(const float32x4& x)  { return loop4(Ops<float>::abs, x); }
  static inline float32x4 sqr(const float32x4& x)  { return _mm_mul_ps(x, x); }
  static inline float32x4 sqrt(const float32x4& x) { return _mm_sqrt_ps(x); }
  static inline float32x4 neg(const float32x4& x)  { return sub(0.f, x); }

  // @TODO: get rid of loop4 with proper intrisics
  static inline float32x4 sgn(const float32x4& x)  { return loop4(Ops<float>::sgn, x); }

  static inline float32x4 round(const float32x4& x)  { return _mm_round_ps(x, _MM_FROUND_TO_NEAREST_INT); }
  static inline float32x4 floor(const float32x4& x)  { return _mm_floor_ps(x); }
  static inline float32x4 ceil(const float32x4& x)   { return _mm_ceil_ps(x); }

  static inline float32x4 add(const float32x4& x, const float32x4& y) { return _mm_add_ps(x, y); }
  static inline float32x4 sub(const float32x4& x, const float32x4& y) { return _mm_sub_ps(x, y); }
  static inline float32x4 mul(const float32x4& x, const float32x4& y) { return _mm_mul_ps(x, y); }
  static inline float32x4 div(const float32x4& x, const float32x4& y) { return _mm_div_ps(x, y); }

  static inline float32x4 max(const float32x4& x, const float32x4& y) { return _mm_max_ps(x, y); }
  static inline float32x4 min(const float32x4& x, const float32x4& y) { return _mm_min_ps(x, y); }
  static inline float32x4 pow(const float32x4& x, const float32x4& y) { return exp(mul(y, log(x))); }

  // @TODO: get rid of loop4 with proper intrisics
  static inline float32x4 negate(float32x4& x)  { return loop4(Ops<float>::negate, x); }

  static inline float32x4 eq(const float32x4& x, const float32x4& y)   { return loop4(Ops<float>::eq, x, y); }
  static inline float32x4 neq(const float32x4& x, const float32x4& y)  { return loop4(Ops<float>::neq, x, y); }
  static inline float32x4 gt(const float32x4& x, const float32x4& y)   { return loop4(Ops<float>::gt, x, y); }
  static inline float32x4 lt(const float32x4& x, const float32x4& y)   { return loop4(Ops<float>::lt, x, y); }
  static inline float32x4 geq(const float32x4& x, const float32x4& y)  { return loop4(Ops<float>::geq, x, y); }
  static inline float32x4 leq(const float32x4& x, const float32x4& y)  { return loop4(Ops<float>::leq, x, y); }
  static inline float32x4 and_(const float32x4& x, const float32x4& y) { return loop4(Ops<float>::and_, x, y); } // 'and' is used by gcc
  static inline float32x4 or_(const float32x4& x, const float32x4& y)  { return loop4(Ops<float>::or_, x, y); } // 'or' is used by gcc

  // Neural Networks specific functions
  // @TODO: this is unsafe
  static inline float32x4 sigmoid(const float32x4& x) {
    float32x4 e = exp(x);
    return div(e, add(1.f, e));
  }

  // // Neural Networks specific functions
  // static HOST_DEVICE_INLINE float sigmoid(const float& x) {
  //   return x > 0 ? (1.f / (1.f + exp(-x))) : (exp(x) / (1.f + exp(x)));
  // }

  static inline float32x4 logaddexp(const float32x4& x, const float32x4& y)  { return loop4(Ops<float>::logaddexp, x, y); }

  static inline float32x4 clip(const float32x4& x, const float32x4& y)  { return loop4(Ops<float>::clip, x, y); }
  static inline float32x4 bump(const float32x4& x, const float32x4& y)  { return loop4(Ops<float>::bump, x, y); }

  static inline float32x4 relu(const float32x4& x)  { return max(0.f, x); }

  static inline float32x4 reluBack(const float32x4& x)  { return loop4(Ops<float>::reluBack, x); }
  static inline float32x4 prelu(const float32x4& x, const float32x4& y)  { return loop4(Ops<float>::prelu, x, y); }
  static inline float32x4 preluBack(const float32x4& x, const float32x4& y)  { return loop4(Ops<float>::preluBack, x, y); }

  static inline float32x4 if_then_else(const float32x4& x, const float32x4& y, const float32x4& z) { return loop4(Ops<float>::if_then_else, x, y, z);  }

  static inline Single sumReduce(const float32x4& x) {
    Single sum = 0;
    for(int i = 0; i < 4; ++i)
      sum = Ops<Single>::add(sum, x[i]);
    return sum;
  }

  static inline Single maxReduce(const float32x4& x) {
    Single maxs = x[0];
    for(int i = 1; i < 4; ++i)
      maxs = Ops<Single>::max(maxs, x[i]);
    return maxs;
  }

  static inline Single minReduce(const float32x4& x) {
    Single mins = x[0];
    for(int i = 1; i < 4; ++i)
      mins = Ops<Single>::min(mins, x[i]);
    return mins;
  }


};

} // end namespace functional
} // end namespace marian
#ifdef __AVX__
#include "3rd_party/avx_mathfun.h"

namespace marian {
namespace functional {

//*******************************************************************************************
// Specialization for float32x8 (=__m256, CPU AVX intrisics)
template <>
struct Ops<float32x8> {
  typedef float Single;

  static inline float32x8 loop8(const std::function<float(const float&)>& f, const float32x8& x) {
    float32x8 out;
    for(int i = 0; i < 8; i++)
      ((float*)&out)[i] = f(((const float*)&x)[i]);
    return out;
  }

  static inline float32x8 loop8(const std::function<float(const float&, const float&)>& f, const float32x8& x, const float32x8& y) {
    float32x8 out;
    for(int i = 0; i < 8; i++)
      ((float*)&out)[i] = f(((const float*)&x)[i], ((const float*)&y)[i]);
    return out;
  }

  static inline float32x8 loop8(const std::function<float(const float&, const float&, const float&)>& f, const float32x8& x, const float32x8& y, const float32x8& z) {
    float32x8 out;
    for(int i = 0; i < 8; i++)
      ((float*)&out)[i] = f(((const float*)&x)[i], ((const float*)&y)[i], ((const float*)&z)[i]);
    return out;
  }

  static inline float32x8 tanh(const float32x8& x) { // ( e^x - e^-x )/( e^x + e^-x )
    float32x8 e2x = exp(mul(2.f, x));
    return div(sub(e2x, 1.f), add(e2x, 1.f));
  }

  static inline float32x8 sin(const float32x8& x) { return sin256_ps(x); }
  static inline float32x8 cos(const float32x8& x) { return cos256_ps(x); }
  static inline float32x8 tan(const float32x8& x) { return div(sin(x), cos(x)); } // @TODO: use sincos256_ps
  static inline float32x8 log(const float32x8& x) { return log256_ps(x); }
  static inline float32x8 exp(const float32x8& x) { return exp256_ps(x); }

  // @TODO: get rid of loop8 with proper intrisics
  static inline float32x8 abs(const float32x8& x)  { return loop8(Ops<float>::abs, x); }
  static inline float32x8 sqr(const float32x8& x)  { return _mm256_mul_ps(x, x); }
  static inline float32x8 sqrt(const float32x8& x) { return _mm256_sqrt_ps(x); }
  static inline float32x8 neg(const float32x8& x)  { return sub(0.f, x); }

  // @TODO: get rid of loop8 with proper intrisics
  static inline float32x8 sgn(const float32x8& x)  { return loop8(Ops<float>::sgn, x); }

  static inline float32x8 round(const float32x8& x)  { return _mm256_round_ps(x, _MM_FROUND_TO_NEAREST_INT); }
  static inline float32x8 floor(const float32x8& x)  { return _mm256_floor_ps(x); }
  static inline float32x8 ceil(const float32x8& x)   { return _mm256_ceil_ps(x); }

  static inline float32x8 add(const float32x8& x, const float32x8& y) { return _mm256_add_ps(x, y); }
  static inline float32x8 sub(const float32x8& x, const float32x8& y) { return _mm256_sub_ps(x, y); }
  static inline float32x8 mul(const float32x8& x, const float32x8& y) { return _mm256_mul_ps(x, y); }
  static inline float32x8 div(const float32x8& x, const float32x8& y) { return _mm256_div_ps(x, y); }

  static inline float32x8 max(const float32x8& x, const float32x8& y) { return _mm256_max_ps(x, y); }
  static inline float32x8 min(const float32x8& x, const float32x8& y) { return _mm256_min_ps(x, y); }
  static inline float32x8 pow(const float32x8& x, const float32x8& y) { return exp(mul(y, log(x))); }

  // @TODO: get rid of loop8 with proper intrisics
  static inline float32x8 negate(float32x8& x)  { return loop8(Ops<float>::negate, x); }

  static inline float32x8 eq(const float32x8& x, const float32x8& y)   { return loop8(Ops<float>::eq, x, y); }
  static inline float32x8 neq(const float32x8& x, const float32x8& y)  { return loop8(Ops<float>::neq, x, y); }
  static inline float32x8 gt(const float32x8& x, const float32x8& y)   { return loop8(Ops<float>::gt, x, y); }
  static inline float32x8 lt(const float32x8& x, const float32x8& y)   { return loop8(Ops<float>::lt, x, y); }
  static inline float32x8 geq(const float32x8& x, const float32x8& y)  { return loop8(Ops<float>::geq, x, y); }
  static inline float32x8 leq(const float32x8& x, const float32x8& y)  { return loop8(Ops<float>::leq, x, y); }
  static inline float32x8 and_(const float32x8& x, const float32x8& y) { return loop8(Ops<float>::and_, x, y); } // 'and' is used by gcc
  static inline float32x8 or_(const float32x8& x, const float32x8& y)  { return loop8(Ops<float>::or_, x, y); } // 'or' is used by gcc


  // Neural Networks specific functions
  // @TODO: this is unsafe
  static inline float32x8 sigmoid(const float32x8& x) {
    float32x8 e = exp(x);
    return div(e, add(1.f, e));
  }

  static inline float32x8 logaddexp(const float32x8& x, const float32x8& y)  { return loop8(Ops<float>::logaddexp, x, y); }

  static inline float32x8 clip(const float32x8& x, const float32x8& y)  { return loop8(Ops<float>::clip, x, y); }
  static inline float32x8 bump(const float32x8& x, const float32x8& y)  { return loop8(Ops<float>::bump, x, y); }

  static inline float32x8 relu(const float32x8& x)  { return max(0.f, x); }

  static inline float32x8 reluBack(const float32x8& x)  { return loop8(Ops<float>::reluBack, x); }
  static inline float32x8 prelu(const float32x8& x, const float32x8& y)  { return loop8(Ops<float>::prelu, x, y); }
  static inline float32x8 preluBack(const float32x8& x, const float32x8& y)  { return loop8(Ops<float>::preluBack, x, y); }

  static inline float32x8 if_then_else(const float32x8& x, const float32x8& y, const float32x8& z) { return loop8(Ops<float>::if_then_else, x, y, z);  }

  static inline Single sumReduce(const float32x8& x) {
    Single sum = 0;
    for(int i = 0; i < 8; ++i)
      sum = Ops<Single>::add(sum, x[i]);
    return sum;
  }

  static inline Single maxReduce(const float32x8& x) {
    Single maxs = x[0];
    for(int i = 1; i < 8; ++i)
      maxs = Ops<Single>::max(maxs, x[i]);
    return maxs;
  }

  static inline Single minReduce(const float32x8& x) {
    Single mins = x[0];
    for(int i = 1; i < 8; ++i)
      mins = Ops<Single>::min(mins, x[i]);
    return mins;
  }
};

} // end namespace functional
} // end namespace marian
#endif
#endif // of "#ifndef __CUDACC__"

#ifdef __CUDACC__
#if COMPILE_FP16

namespace marian {
namespace functional {

// Specialization for half
template <>
struct Ops<half> {

  static DEVICE_INLINE half sin(const half& x)  { return hsin(x); }
  static DEVICE_INLINE half cos(const half& x)  { return hcos(x); }
  static DEVICE_INLINE half tan(const half& x)  { return hsin(x) / hcos(x); }
  static DEVICE_INLINE half log(const half& x)  { return hlog(x); }
  static DEVICE_INLINE half exp(const half& x)  { return hexp(x); }
  static DEVICE_INLINE half sqr(const half& x)  { return x * x; }
  static DEVICE_INLINE half sqrt(const half& x) { return hsqrt(x); }
  static DEVICE_INLINE half neg(const half& x)  { return -x; }

#if CUDA_VERSION < 11000
  static DEVICE_INLINE half abs(const half& x)  { return fabs((float)x); }
#else
  static DEVICE_INLINE half abs(const half& x)  { return __habs(x); }
#endif

  static DEVICE_INLINE half sgn(const half& x)  { half zero = 0.f; return (zero < x) - (x < zero); } // @TODO half has this information somewhere in the struct, right?

  static DEVICE_INLINE half round(const half& x)  { return hrint(x); }
  static DEVICE_INLINE half floor(const half& x)  { return hfloor(x); }
  static DEVICE_INLINE half ceil(const half& x)   { return hceil(x); } 

  static DEVICE_INLINE half add(const half& x, const half& y)  { return x + y; }
  static DEVICE_INLINE half sub(const half& x, const half& y)  { return x - y; }
  static DEVICE_INLINE half mul(const half& x, const half& y)  { return x * y; }
  static DEVICE_INLINE half div(const half& x, const half& y)  { return x / y; }

  static DEVICE_INLINE half max(const half& x, const half& y)  { return x < y ? y : x; }
  static DEVICE_INLINE half min(const half& x, const half& y)  { return x < y ? x : y; }
  static DEVICE_INLINE half pow(const half& x, const half& y)  { return exp(y * log(x)); }

  static DEVICE_INLINE half negate(const half& x)  { return !(bool)x; }
  static DEVICE_INLINE half eq(const half& x, const half& y)   { return x == y; }
  static DEVICE_INLINE half neq(const half& x, const half& y)  { return x != y; }
  static DEVICE_INLINE half gt(const half& x, const half& y)   { return x > y;  }
  static DEVICE_INLINE half lt(const half& x, const half& y)   { return x < y;  }
  static DEVICE_INLINE half geq(const half& x, const half& y)  { return x >= y; }
  static DEVICE_INLINE half leq(const half& x, const half& y)  { return x <= y; }
  static DEVICE_INLINE half and_(const half& x, const half& y) { return x && y; } // 'and' is used by gcc
  static DEVICE_INLINE half or_(const half& x, const half& y)  { return x || y; } // 'or' is used by gcc

  // Neural Networks specific functions
  static DEVICE_INLINE half sigmoid(const half& x) {
    const half zero = 0.f;
    const half one  = 1.f;
    return x > zero ? (one / (one + exp(-x))) : (exp(x) / (one + exp(x))); // safe sigmoid
  }

  static DEVICE_INLINE half tanh(const half& x) {
    // tanh(x) = 2 * sigmoid(2 * x) - 1
    const half one = 1.f;
    const half two = 2.f;
    return two * sigmoid(two * x) - one; // safe sigmoid => safe tanh
  }

  static DEVICE_INLINE half log1p(const half& x) {
    return log(x + (half)1.f); // probably acceptable loss of precision, it's half anyway
  }

  static DEVICE_INLINE half logaddexp(const half& x, const half& y) {
    // Note: This may not be ideal for CUDA; cf. CNTK implementation
    return x < y ? (y + log1p(exp(x - y))) : (x + log1p(exp(y - x)));
  }

  static DEVICE_INLINE half clip(const half& x, const half& y)  { return abs(x) >= y ? sgn(x) * y : x; }

  // derivative of Clip, cut-off function
  static DEVICE_INLINE half bump(const half& x, const half& y)  {
    const half zero = 0.f;
    const half one =  1.f;
    return abs(x) >= y ? zero : one;
  }
  static DEVICE_INLINE half relu(const half& x) {
    const half zero = 0.f;
    return max(x, zero);
  }
  static DEVICE_INLINE half reluBack(const half& x) {
    const half zero = 0.f;
    return geq(x, zero);
  }

  static DEVICE_INLINE half prelu(const half& x, const half& y)     {
    const half zero = 0.f;
    return x > zero ? x : x * y;
  }

  static DEVICE_INLINE half preluBack(const half& x, const half& y) {
    const half zero = 0.f;
    const half one =  1.f;
    return x > zero ? one : y;
  }

  static DEVICE_INLINE half if_then_else(const half& x, const half& y, const half& z) { return x ? y : z; }

  static DEVICE_INLINE half sumReduce(const half& x) { return x; }
  static DEVICE_INLINE half maxReduce(const half& x) { return x; }
  static DEVICE_INLINE half minReduce(const half& x) { return x; }

};

// Specialization for halfx2
template <>
struct Ops<halfx2> {

  static DEVICE_INLINE halfx2 sin(const halfx2& x)  { return h2sin(x); }
  static DEVICE_INLINE halfx2 cos(const halfx2& x)  { return h2cos(x); }
  static DEVICE_INLINE halfx2 tan(const halfx2& x)  { return h2sin(x) / h2cos(x); }
  static DEVICE_INLINE halfx2 log(const halfx2& x)  { return h2log(x); }
  static DEVICE_INLINE halfx2 exp(const halfx2& x)  { return h2exp(x); }
  static DEVICE_INLINE halfx2 sqr(const halfx2& x)  { return __hmul2(x, x); }
  static DEVICE_INLINE halfx2 sqrt(const halfx2& x) { return h2sqrt(x); }
  static DEVICE_INLINE halfx2 neg(const halfx2& x)  { return __hneg2(x); }

#if CUDA_VERSION < 11000
  static DEVICE_INLINE halfx2 abs(const halfx2& x)  { return {Ops<half>::abs(x[0]), Ops<half>::abs(x[1])}; }
#else
  static DEVICE_INLINE halfx2 abs(const halfx2& x)  { return __habs2(x); }
#endif
  
  static DEVICE_INLINE halfx2 sgn(const halfx2& x)  { halfx2 zero(0.f, 0.f); return __hsub2(__hlt2(zero, x), __hlt2(x, zero)); }

  static DEVICE_INLINE halfx2 round(const halfx2& x)  { return h2rint(x); }
  static DEVICE_INLINE halfx2 floor(const halfx2& x)  { return h2floor(x); }
  static DEVICE_INLINE halfx2 ceil(const halfx2& x)   { return h2ceil(x); } 

  static DEVICE_INLINE halfx2 add(const halfx2& x, const halfx2& y)  { return __hadd2(x, y); }
  static DEVICE_INLINE halfx2 sub(const halfx2& x, const halfx2& y)  { return __hsub2(x, y); }
  static DEVICE_INLINE halfx2 mul(const halfx2& x, const halfx2& y)  { return __hmul2(x, y); }
  static DEVICE_INLINE halfx2 div(const halfx2& x, const halfx2& y)  { return __h2div(x, y); }

  static DEVICE_INLINE halfx2 pow(const halfx2& x, const halfx2& y)  { return exp(mul(y, log(x))); }

  // Neural Networks specific functions
  // @TODO: this is unsafe
  static DEVICE_INLINE halfx2 sigmoid(const halfx2& x) {
    halfx2 e = exp(x);
    halfx2 one(1.f, 1.f);
    return div(e, add(one, e));
  }

  static DEVICE_INLINE halfx2 tanh(const halfx2& x) {
    // tanh(x) = 2 * sigmoid(2 * x) - 1
    const halfx2 one(1.f);
    const halfx2 two(2.f);
    return sub(mul(two, sigmoid(mul(two, x))), one); 
  }

  static DEVICE_INLINE halfx2 max(const halfx2& x, const halfx2& y)  { 
    return { Ops<half>::max(x[0], y[0]), Ops<half>::max(x[1], y[1]) }; 
  }

  static DEVICE_INLINE halfx2 min(const halfx2& x, const halfx2& y)  { 
    return { Ops<half>::min(x[0], y[0]), Ops<half>::min(x[1], y[1]) }; 
  }
  
  static DEVICE_INLINE halfx2 negate(const halfx2& x)  { return { Ops<half>::negate(x[0]), Ops<half>::negate(x[1]) }; }
  static DEVICE_INLINE halfx2 eq(const halfx2& x, const halfx2& y)   { return __heq2(x, y); }
  static DEVICE_INLINE halfx2 neq(const halfx2& x, const halfx2& y)  { return __hne2(x, y); }
  static DEVICE_INLINE halfx2 gt(const halfx2& x, const halfx2& y)   { return __hgt2(x, y); }
  static DEVICE_INLINE halfx2 lt(const halfx2& x, const halfx2& y)   { return __hlt2(x, y); }
  static DEVICE_INLINE halfx2 geq(const halfx2& x, const halfx2& y)  { return __hge2(x, y); }
  static DEVICE_INLINE halfx2 leq(const halfx2& x, const halfx2& y)  { return __hle2(x, y); }
  static DEVICE_INLINE halfx2 and_(const halfx2& x, const halfx2& y) { return {x[0] && y[0], x[1] && y[1]}; } // 'and' is used by gcc
  static DEVICE_INLINE halfx2 or_(const halfx2& x, const halfx2& y)  { return {x[0] || y[0], x[1] || y[1]}; } // 'or' is used by gcc
  
  static DEVICE_INLINE halfx2 log1p(const halfx2& x) {
    return log(add(x, halfx2(1.f))); // probably acceptable loss of precision, it's half anyway
  }

  static DEVICE_INLINE halfx2 if_then_else(const halfx2& x, const halfx2& y, const halfx2& z) { 
    return {x[0] ? y[0] : z[0], x[1] ? y[1] : z[1]}; 
  }

  static DEVICE_INLINE halfx2 logaddexp(const halfx2& x, const halfx2& y) {
    // Note: This may not be ideal for CUDA; cf. CNTK implementation
    auto a = add(y, log1p(exp(sub(x, y)))); 
    auto b = add(x, log1p(exp(sub(y, x))));
    return if_then_else(lt(x, y), a, b);
  }

  static DEVICE_INLINE halfx2 clip(const halfx2& x, const halfx2& y)  { 
    return if_then_else(geq(abs(x), y), mul(sgn(x), y), x); 
  }

  // derivative of Clip, cut-off function
  static DEVICE_INLINE halfx2 bump(const halfx2& x, const halfx2& y)  {
    const halfx2 zero(0.f);
    const halfx2 one(1.f);
    return if_then_else(geq(abs(x), y), zero, one);
  }
  static DEVICE_INLINE halfx2 relu(const halfx2& x) {
    const halfx2 zero(0.f);
    return max(x, zero);
  }
  static DEVICE_INLINE halfx2 reluBack(const halfx2& x) {
    const halfx2 zero(0.f);
    return geq(x, zero);
  }

  static DEVICE_INLINE halfx2 prelu(const halfx2& x, const halfx2& y)     {
    const halfx2 zero(0.f);
    return if_then_else(gt(x, zero), x , mul(x, y));
  }

  static DEVICE_INLINE halfx2 preluBack(const halfx2& x, const halfx2& y) {
    const halfx2 zero(0.f);
    const halfx2 one(1.f);
    return if_then_else(gt(x, zero), one, y);
  }

};

} // end namespace functional
} // end namespace marian

#endif
#endif

//*******************************************************************************************

#include "functional/defs.h"
#include "functional/predicates.h"

namespace marian {
namespace functional {

UNARY(Tanh,    tanh,       Ops<ElementType>::tanh(x));
UNARY(Sin,     sin,        Ops<ElementType>::sin(x));
UNARY(Cos,     cos,        Ops<ElementType>::cos(x));
UNARY(Tan,     tan,        Ops<ElementType>::tan(x));
UNARY(Log,     log,        Ops<ElementType>::log(x));
UNARY(Exp,     exp,        Ops<ElementType>::exp(x));
UNARY(Abs,     abs,        Ops<ElementType>::abs(x));
UNARY(Sqr,     sqr,        Ops<ElementType>::sqr(x));
UNARY(Sqrt,    sqrt,       Ops<ElementType>::sqrt(x));
UNARY(Neg,     operator-,  Ops<ElementType>::neg(x));
UNARY(Sgn,     sgn,        Ops<ElementType>::sgn(x));

UNARY(Round,   round,      Ops<ElementType>::round(x));
UNARY(Floor,   floor,      Ops<ElementType>::floor(x));
UNARY(Ceil,    ceil,       Ops<ElementType>::ceil(x));

BINARY(Plus,   operator+,  Ops<ElementType>::add(x, y));
BINARY(Minus,  operator-,  Ops<ElementType>::sub(x, y));
BINARY(Mult,   operator*,  Ops<ElementType>::mul(x, y));
BINARY(Div,    operator/,  Ops<ElementType>::div(x, y));
BINARY(Max,    max,        Ops<ElementType>::max(x, y));
BINARY(Min,    min,        Ops<ElementType>::min(x, y));
UNARY(Negate,  operator!,  Ops<ElementType>::negate(x));
BINARY(Eq,     operator==, Ops<ElementType>::eq(x, y));
BINARY(NEq,    operator!=, Ops<ElementType>::neq(x, y));
BINARY(Gt,     operator>,  Ops<ElementType>::gt(x, y));
BINARY(Lt,     operator<,  Ops<ElementType>::lt(x, y));
BINARY(Geq,    operator>=, Ops<ElementType>::geq(x, y));
BINARY(Leq,    operator<=, Ops<ElementType>::leq(x, y));
BINARY(And,    operator&&, Ops<ElementType>::and_(x, y));
BINARY(Or,     operator||, Ops<ElementType>::or_(x, y));
BINARY(Pow,    pow,        Ops<ElementType>::pow(x, y));

TERNARY(IfThenElse, if_then_else, Ops<ElementType>::if_then_else(x, y, z));

// Neural Networks specific functions
BINARY(Clip,       clip,      Ops<ElementType>::clip(x, y));
// derivative of Clip, cut-off function
BINARY(Bump,       bump,      Ops<ElementType>::bump(x, y));

UNARY(Sigmoid,     sigmoid,   Ops<ElementType>::sigmoid(x));
BINARY(LogAddExp,  logaddexp, Ops<ElementType>::logaddexp(x, y));
UNARY(sReLU,       ReLU,      Ops<ElementType>::relu(x));
UNARY(sReLUBack,   ReLUback,  Ops<ElementType>::reluBack(x));
BINARY(sPReLU,     PReLU,     Ops<ElementType>::prelu(x, y));
BINARY(sPReLUBack, PReLUback, Ops<ElementType>::preluBack(x, y));

} // end namespace functional
} // end namespace marian
