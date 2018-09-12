#pragma once

#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "common/logging.h"
#include "shape.h"

#define THREAD_GUARD(body) std::thread([&]() { body; }).join()
#define NodeOp(op) [=]() { op; }

namespace marian {

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
}  // namespace marian

#include "keywords.h"

namespace marian {

enum class DeviceType : size_t { gpu = 0, cpu = 1 };

struct DeviceId {
  size_t no{0};
  DeviceType type{DeviceType::gpu};

  DeviceId() : no{0}, type{DeviceType::gpu} {}
  DeviceId(size_t no_, DeviceType type_) : no(no_), type(type_) {}

  friend std::ostream& operator<<(std::ostream& out, DeviceId deviceId) {
    out << (deviceId.type == DeviceType::gpu ? "gpu" : "cpu") << deviceId.no;
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

class LexProbs;

/**
 * @brief Defines a set of keywords.
 *
 * Each invocation of the KEY(name, value_type) macro
 *    will result in the creation of an instance of the Keyword class.
 */
namespace keywords {
KEY(axis, int);
KEY(shape, Shape);
KEY(value, float);
KEY(fixed, bool);
//KEY(prefix, std::string); // (conflicts with local variables named prefix)
KEY(final, bool);
KEY(output_last, bool);
KEY(mask, Expr);
KEY(dropout_prob, float);
KEY(init, std::function<void(Tensor)>);

KEY(eta, float);
KEY(beta1, float);
KEY(beta2, float);
KEY(eps, float);
KEY(optimizer, Ptr<OptimizerBase>);
KEY(clip, Ptr<ClipperBase>);
KEY(batch_size, int);
KEY(normalize, bool);
KEY(inference, bool);
KEY(skip, bool);
KEY(skip_first, bool);
KEY(coverage, Expr);
KEY(max_epochs, int);
KEY(valid, Ptr<RunBase>);
KEY(lex_probs, Ptr<LexProbs>);
}  // namespace keywords

const float NEMATUS_LN_EPS = 1e-5f;
}  // namespace marian
