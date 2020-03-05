#pragma once

#include "common/logging.h"
#include "common/shape.h"
#include "common/intrusive_ptr.h"

#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

// The macro MAYBE_UNUSED is used to selectively disable
// unused-variable warnings. C++17 defines the attribute
// [[maybe_unused]], but I don't think we're at C++17 yet. We can add it when we reach C++17.
// The compilers gcc and clang (and maybe others) define
// __has_attribute and support __attribute__(unused) in C++11,
#if defined __has_attribute
#  if __has_attribute(unused)
#    define MAYBE_UNUSED __attribute__((unused))
#  else
#    define MAYBE_UNUSED
#  endif
#endif



#define THREAD_GUARD(body) [&]() { body; }() // test if THREAD_GUARD is neccessary, remove if no problems occur.
#define NodeOp(op) [=]() { op; }

// helper macro to disable optimization (gcc only)
// To use this, just insert DONT_OPTIMIZE right before the function definition
// (e.g. where the "static" keyword would go).
#ifdef __GNUC__
#define DONT_OPTIMIZE __attribute__((optimize("O0")))
#else
#define DONT_OPTIMIZE // silently ignore on Visual Studio, where this is less of a problem
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
Ptr<T> New(Args&&... args) {
  return Ptr<T>(new T(std::forward<Args>(args)...));
}

template <class T>
Ptr<T> New(Ptr<T> p) {
  return Ptr<T>(p);
}

/** @brief Creates InstrusivePtr of any type, passes all arguments to any available
 * constructor */
template <class T, typename... Args>
IPtr<T> INew(Args&&... args) {
  return IPtr<T>(new T(std::forward<Args>(args)...));
}

template <class T>
IPtr<T> INew(Ptr<T> p) {
  return IPtr<T>(p);
}

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
