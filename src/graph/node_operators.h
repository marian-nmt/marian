#pragma once

#include "graph/node.h"
#include "graph/node_initializers.h"
#include "tensors/tensor.h"

namespace marian {

struct ConstantNode : public Node {
  ConstantNode(Ptr<ExpressionGraph> graph, const Shape& shape, const NodeInitializer& init)
      : Node(graph, shape),
        init_(new NodeInitializer(init)),
        initialized_(false) {

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

  virtual bool equal(Expr node) { return this == node.get(); }

private:
  UPtr<NodeInitializer> init_;
  bool initialized_;
};

struct ParamNode : public Node {
  ParamNode(Ptr<ExpressionGraph> graph, const Shape& shape, const NodeInitializer& init, bool fixed = false)
      : Node(graph, shape),
        init_(new NodeInitializer(init)),
        initialized_(false) {

    setTrainable(!fixed);
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

  virtual bool equal(Expr node) { return name() == node->name(); }

private:
  UPtr<NodeInitializer> init_;
  bool initialized_;
};
}
