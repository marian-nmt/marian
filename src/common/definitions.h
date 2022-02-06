#pragma once

#include "common/logging.h"
#include "common/shape.h"
#include "common/intrusive_ptr.h"

#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#define THREAD_GUARD(body) [&]() { body; }() // test if THREAD_GUARD is necessary, remove if no problems occur.
#define NodeOp(op) [=]() { op; }

// helper macro to disable optimization (gcc only)
// To use this, just insert DONT_OPTIMIZE right before the function definition
// (e.g. where the "static" keyword would go).
#ifdef __GNUC__
#define DONT_OPTIMIZE __attribute__((optimize("O0")))
#else
#define DONT_OPTIMIZE // silently ignore on Visual Studio, where this is less of a problem
#endif

// Use these macros to enable faster floating-point math. Put them around one
// or more functions.
//
// Usage:
// MARIAN_FFAST_MATH_BEGIN
// void LayerNormalization(float *arg) { *arg += 1.0; }
// void SomethingElse() {}
// MARIAN_FFAST_MATH_END
//
// ffast-math allows the compiler to assume associative arithmetic and finite
// values.
//
// Associative arithmetic is particularly important to vectorize i.e. a sum:
//   for (const float f : range) sum += f;
// Without ffast-math, the sum will be done one value at a time.  On x86 it
// still uses vector math, but only uses the first slot and wastes the rest.
//
// With ffast-math, the compiler can sum in batches of 4, 8, or 16 floats.
// Also, it can run multiple adds in parallel e.g. vaddps has latency 4 and
// throughput 0.5 on Skylake so multiple vector adds can run at once.
//
// On average, a vectorized sum is more numerically stable because it sums in
// batches. Vectorized floats can still produce NaNs and infs (remember even
// scalar operations are implemented with vector instructions).
//
// Allowing the compiler to assume finite values means functions like isnan or
// isinf do not work as expected. Do not enable this for a function that
// depends upon fully standard float behavior.
//
// It can also change the sign of zeros.
//
// Fast math also makes results more architecture dependent because different
// register widths mean different results. They also depend on the compiler
// and compiler version more. For example, clang <= 10 does not support the
// float_control pragma below so it will still be conservative.
//
// There is a more conservative option for just associativity:
// llvm introduced "#pragma clang fp reassociate" that goes inside a function.
// However, llvm <11 considers that pragma an error so we'd need some ugly
// version test (which they don't recommend) or a compilation test.  Moreover,
// it has to be in the function to keep scope.
// gcc supports "-fassociative-math" that has to be outside a function.
// I didn't find a MSVC equivalent.
#if defined(_MSC_VER)
#define MARIAN_FFAST_MATH_BEGIN __pragma(float_control(precise, off, push))
#define MARIAN_FFAST_MATH_END __pragma(float_control(pop))
#elif defined(__clang__)
#define MARIAN_FFAST_MATH_BEGIN _Pragma("float_control(precise, off, push)")
#define MARIAN_FFAST_MATH_END _Pragma("float_control(pop)")
#elif defined(__GNUC__)
// Also available as __attribute__((optimize("-ffast-math"))) but done as pragmas for consistency
#define MARIAN_FFAST_MATH_BEGIN _Pragma("GCC push_options") _Pragma("GCC optimize(\"-ffast-math\")")
#define MARIAN_FFAST_MATH_END _Pragma("GCC pop_options")
#endif

namespace marian {

// Type to be used for all index types, e.g. for integer tensors for rows operator.
// size_t would seem to be the natural choice over uint32_t but has usually 8 bytes
// while uint32_t has 4 bytes. This type will be often exchanged between CPU and GPU.
// This minimizes bandwith at little cost.
typedef uint32_t IndexType;

// @TODO: come up with better short name. "I..." stands for interface now. Here it stands
// for "intrusive". Not a good overlap.
template <class T>
using IPtr = IntrusivePtr<T>;

template <class T>
using UPtr = std::unique_ptr<T>;

// @TODO: come up with better short name. "I..." stands for interface now.
template <class T>
using IWeak = T*;

template <class T>
using Ptr = std::shared_ptr<T>;

template <class T>
using Weak = std::weak_ptr<T>;

/** @brief Creates shared_ptr of any type, passes all arguments to any available
 * constructor */
template <class T, typename... Args>
inline Ptr<T> New(Args&&... args) {
  return std::make_shared<T>(std::forward<Args>(args)...);
}

template <class T>
inline Ptr<T> New(Ptr<T> p) {
  return Ptr<T>(p);
}

/** @brief Creates InstrusivePtr of any type, passes all arguments to any available
 * constructor */
template <class T, typename... Args>
inline IPtr<T> INew(Args&&... args) {
  return IPtr<T>(new T(std::forward<Args>(args)...));
}

template <class T>
inline IPtr<T> INew(Ptr<T> p) {
  return IPtr<T>(p);
}

/// enum class DeviceType: defines which device is used for computation
enum class DeviceType : size_t { gpu = 0, cpu = 1 };

struct DeviceId {
  size_t no{0};
  DeviceType type{DeviceType::gpu};

  DeviceId() : no{0}, type{DeviceType::gpu} {}
  DeviceId(size_t no_, DeviceType type_) : no(no_), type(type_) {}

  std::string typeAsString() const {
    return (type == DeviceType::gpu ? "gpu" : "cpu");
  }

  operator std::string() const {
    return typeAsString() + std::to_string(no);
  }

  friend std::ostream& operator<<(std::ostream& out, DeviceId deviceId) {
    out << std::string(deviceId);
    return out;
  }

  friend bool operator==(DeviceId id1, DeviceId id2) {
    return id1.no == id2.no && id1.type == id2.type;
  }
  friend bool operator!=(DeviceId id1, DeviceId id2) { return !(id1 == id2); }
};

// predefine a couple of devices for easier manual use
const DeviceId CPU0{0, DeviceType::cpu};
const DeviceId CPU1{1, DeviceType::cpu};
const DeviceId CPU2{2, DeviceType::cpu};
const DeviceId CPU3{3, DeviceType::cpu};
const DeviceId CPU4{4, DeviceType::cpu};
const DeviceId CPU5{5, DeviceType::cpu};
const DeviceId CPU6{6, DeviceType::cpu};
const DeviceId CPU7{7, DeviceType::cpu};

const DeviceId GPU0{0, DeviceType::gpu};
const DeviceId GPU1{1, DeviceType::gpu};
const DeviceId GPU2{2, DeviceType::gpu};
const DeviceId GPU3{3, DeviceType::gpu};
const DeviceId GPU4{4, DeviceType::gpu};
const DeviceId GPU5{5, DeviceType::gpu};
const DeviceId GPU6{6, DeviceType::gpu};
const DeviceId GPU7{7, DeviceType::gpu};

// These are many small objects, hence use IntrusivePtr
class TensorBase;
typedef IPtr<TensorBase> Tensor;

// These are many small objects, hence use IntrusivePtr
template <class DataType>
class Chainable;
typedef IPtr<Chainable<Tensor>> Expr;

class OptimizerBase;
typedef Ptr<OptimizerBase> OptimizerBasePtr;

class ClipperBase;
typedef Ptr<ClipperBase> ClipperBasePtr;

class RunBase;
typedef Ptr<RunBase> RunBasePtr;


const float NEMATUS_LN_EPS = 1e-5f;
}  // namespace marian
