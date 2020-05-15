#pragma once

#include "graph/node.h"
#include "graph/node_initializers.h"
#include "tensors/tensor.h"

namespace marian {

struct ConstantNode : public Node {
  ConstantNode(Ptr<ExpressionGraph> graph,
               const Shape& shape,
               const Ptr<inits::NodeInitializer>& init,
               Type valueType = Type::float32);

  ~ConstantNode() {}

  virtual void allocate() override;
  virtual void init() override;

  const std::string type() override { return "const"; }

  const std::string form() override { return "diamond"; }

  const std::string color() override { return "white"; }

  virtual size_t hash() override {
    size_t seed = util::hash<size_t>()((size_t)this);
    return seed;
  }

  virtual bool equal(Expr node) override { return this == node.get(); }
  virtual void record(Ptr<AutoTunerRecorder>, size_t, bool) override{};

private:
  Ptr<inits::NodeInitializer> init_;
  bool initialized_;
};

struct ParamNode : public Node {
  ParamNode(Ptr<ExpressionGraph> graph,
            const Shape& shape,
            const Ptr<inits::NodeInitializer>& init,
            bool fixed = false);

  ParamNode(Ptr<ExpressionGraph> graph,
            const Shape& shape,
            const Ptr<inits::NodeInitializer>& init,
            Type valueType,
            bool fixed = false);

  ~ParamNode() {}

  virtual void allocate() override {
    ABORT_IF(!val_, "Parameters should be allocated by their graph. Parameter {} was not", name_);
  }

  virtual void init() override;

  const std::string type() override { return "param"; }

  const std::string form() override { return "hexagon"; }

  const std::string color() override { return "orangered"; }

  virtual size_t hash() override {
    size_t seed = util::hash<size_t>()((size_t)this);
    return seed;
  }

  virtual bool equal(Expr node) override { return name() == node->name(); }

  virtual void record(Ptr<AutoTunerRecorder>, size_t, bool) override{};

private:
  Ptr<inits::NodeInitializer> init_;
  bool initialized_;
};
}  // namespace marian
