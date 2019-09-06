#pragma once

#include "functional/defs.h"
#include "functional/operands.h"

namespace marian {
namespace functional {

namespace float2unsigned {
constexpr float abs(float x) {
  return x < 0 ? -x : x;
}

// clang-format off
constexpr int exponent(float x) {
  return abs(x) >= 2 ? exponent(x / 2) + 1 :
                       abs(x) < 1 ? exponent(x * 2) - 1 :
                                    0;
}

constexpr float scalbn(float value, int exponent) {
  return exponent == 0  ? value :
                         exponent > 0 ? scalbn(value * 2, exponent - 1) :
                                        scalbn(value / 2, exponent + 1);
}
// clang-format on

constexpr unsigned mantissa(float x, int exp) {
  // remove hidden 1 and bias the exponent to get integer
  return abs(x) < std::numeric_limits<float>::infinity()
             ? scalbn(scalbn(abs(x), -exp) - 1, 23)
             : 0;
}

constexpr unsigned to_binary(float x, unsigned sign, int exp) {
  return sign * (1u << 31) + (exp + 127) * (1u << 23) + mantissa(x, exp);
}

constexpr unsigned to_binary(float x) {
  return x == 0 ? 0 : to_binary(x, x < 0, exponent(x));
}
}  // namespace float2unsigned

namespace unsigned2float {

constexpr float sign(unsigned i) {
  return (i & (1u << 31)) ? -1.f : 1.f;
}

constexpr int exponent(unsigned i) {
  return int((i >> 23) & 255u) - 127;
}

constexpr float sig(unsigned i, unsigned shift) {
  return ((i >> shift) & 1u) * 1.f / (1u << (23 - shift))
         + (shift > 0 ? sig(i, shift - 1) : 0);
}

constexpr float powr(int exp) {
  return exp > 0 ? 2.f * powr(exp - 1) : 1.f;
}

constexpr float pow(int exp) {
  return exp < 0 ? 1.f / powr(-exp) : powr(exp);
}

constexpr float from_binary(unsigned i) {
  return (1.f + sig(i, 22u)) * pow(exponent(i)) * sign(i);
}
}  // namespace unsigned2float

constexpr unsigned f2i(float x) {
  return float2unsigned::to_binary(x);
}

constexpr float i2f(float x) {
  return unsigned2float::from_binary(x);
}

template <unsigned V>
struct F {
  static constexpr auto value = i2f(V);
  static constexpr auto binary = V;

  template <typename... Args>
  HOST_DEVICE_INLINE constexpr float operator()(Args&&... args) const {
    return value;
  }

  std::string to_string() { return "F<" + std::to_string(value) + ">"; }
};
}  // namespace functional
}  // namespace marian
