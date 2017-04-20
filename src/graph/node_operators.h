#pragma once

#include "graph/node.h"
#include "graph/node_operators_unary.h"
#include "graph/node_operators_binary.h"
#include "kernels/tensor_operators.h"

namespace marian {

struct InputNode : public Node {
  template <typename ...Args>
  InputNode(Args ...args)
  : Node(args...) {
    UTIL_THROW_IF2(!Has(keywords::shape),
                   "Data items require shape information");
    setTrainable(false);
  }

  ~InputNode() {}

  const std::string type() {
    return "input";
  }

  const std::string form() {
    return "circle";
  }

  const std::string color() {
    return "white";
  }

};

struct ConstantNode : public Node {
  template <typename ...Args>
  ConstantNode(Args ...args)
  : Node(args...),
    init_(Get(keywords::init, [](Tensor){ })),
    initialized_(false)
  {
    UTIL_THROW_IF2(!Has(keywords::shape),
                   "Constant items require shape information");
    setTrainable(false);
  }

  ~ConstantNode() {}

  virtual size_t allocate();
  virtual void init();

  const std::string type() {
    return "const";
  }

  const std::string form() {
    return "diamond";
  }

  const std::string color() {
    return "white";
  }

  virtual size_t hash() {
    // @TODO: think of something better for constant nodes
    return boost::hash<size_t>()((size_t)this);
  }

  private:
    std::function<void(Tensor)> init_;
    bool initialized_;
};

struct ParamNode : public Node {
  template <typename ...Args>
  ParamNode(Args ...args)
  : Node(args...),
    init_(Get(keywords::init, [](Tensor){ })),
    initialized_(false)
  {
    UTIL_THROW_IF2(!Has(keywords::shape),
                   "Param items require shape information");
    setTrainable(true);
  }

  ~ParamNode() {}

  virtual size_t allocate();

  virtual void init();

  const std::string type() {
    return "param";
  }

  const std::string form() {
    return "hexagon";
  }

  const std::string color() {
    return "orangered";
  }

  virtual size_t hash() {
    return boost::hash<size_t>()((size_t)this);
  }

  private:
    std::function<void(Tensor&)> init_;
    bool initialized_;
};

}
