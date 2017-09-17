#pragma once

#include "graph/node.h"
#include "tensors/tensor.h"

namespace marian {

struct ConstantNode : public Node {
  template <typename... Args>
  ConstantNode(Args... args)
      : Node(args...),
        init_(Get(keywords::init, [](Tensor) {})),
        initialized_(false) {
    UTIL_THROW_IF2(!Has(keywords::shape),
                   "Constant items require shape information");
    setTrainable(false);
  }

  ~ConstantNode() {}

  virtual size_t allocate();
  virtual void init();

  const std::string type() { return "const"; }

  const std::string form() { return "diamond"; }

  const std::string color() { return "white"; }

  virtual size_t hash() {
    std::size_t seed = boost::hash<std::string>()(name());
    boost::hash_combine(seed, type());
    boost::hash_combine(seed, this);
    return seed;
  }

  virtual bool equal(Expr node) {
    return this == node.get();
  }

private:
  std::function<void(Tensor)> init_;
  bool initialized_;
};

struct ParamNode : public Node {
  template <typename... Args>
  ParamNode(Args... args)
      : Node(args...),
        init_(Get(keywords::init, [](Tensor) {})),
        initialized_(false) {
    UTIL_THROW_IF2(!Has(keywords::shape),
                   "Param items require shape information");
    setTrainable(!Get(keywords::fixed, false));
  }

  ~ParamNode() {}

  virtual size_t allocate();

  virtual void init();

  const std::string type() { return "param"; }

  const std::string form() { return "hexagon"; }

  const std::string color() { return "orangered"; }

  virtual size_t hash() {
    std::size_t seed = boost::hash<std::string>()(name());
    boost::hash_combine(seed, type());
    boost::hash_combine(seed, this);
    return seed;
  }

  virtual bool equal(Expr node) {
    return name() == node->name();
  }

private:
  std::function<void(Tensor&)> init_;
  bool initialized_;
};
}
