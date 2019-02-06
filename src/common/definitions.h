#pragma once

#include "common/logging.h"
#include "shape.h"

#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

//#define THREAD_GUARD(body) std::thread([&]() { body; }).join()
#define THREAD_GUARD(body) [&]() { body; }() // test if THREAD_GUARD is neccessary, remove if no problems occur.
#define NodeOp(op) [=]() { op; }

namespace marian {

// Type to be used for all index types, e.g. for integer tensors for rows operator.
// size_t would seem to be the natural choice over uint32_t but has usually 8 bytes
// while uint32_t has 4 bytes. This type will be often exchanged between CPU and GPU.
// This minimizes bandwith at little cost.
typedef uint32_t IndexType;

template <class T>
using Ptr = std::shared_ptr<T>;

template <class T>
using UPtr = std::unique_ptr<T>;

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

class TensorBase;
typedef Ptr<TensorBase> Tensor;

template <class DataType>
class Chainable;
typedef Ptr<Chainable<Tensor>> Expr;

class OptimizerBase;
typedef Ptr<OptimizerBase> OptimizerBasePtr;

class ClipperBase;
typedef Ptr<ClipperBase> ClipperBasePtr;

class RunBase;
typedef Ptr<RunBase> RunBasePtr;


const float NEMATUS_LN_EPS = 1e-5f;
}  // namespace marian
