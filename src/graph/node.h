#pragma once

#include <iostream>
#include <memory>
#include <thread>

#include "common/keywords.h"
#include "tensors/backend.h"
#include "tensors/tensor.h"

#include "graph/chainable.h"

namespace marian {

/**
 * Main node class for computation graph,
 * implements most common functions demanded by Chainable.
 * Each operation in a computation graph is a node.
 */
class Node : public Chainable<Tensor>,
             public std::enable_shared_from_this<Node> {
protected:
  size_t id_{0};
  size_t edges_{0};
  bool trainable_{true};
  bool destroy_{true};
  bool memoize_{false};

  std::vector<Expr> children_;

  Weak<ExpressionGraph> graph_;
  Shape shape_{1, 1, 1, 1};
  Type value_type_{Type::float32};

  std::string name_{"none"};

  Tensor val_{nullptr};
  Tensor adj_{nullptr};

  bool markedForDebug_{false};
  std::string debugMessage_;

  Ptr<AutoTunerRecorder> recorder_;
  size_t recorderHash_;
  bool recorderStop_;

public:
  Node(Ptr<ExpressionGraph> graph, Shape shape, Type value_type = Type::float32)
      : graph_(graph), shape_(shape), value_type_(value_type) {}

  virtual ~Node() {
    if(destroy_) {
      free();
    }
  }

  virtual float scalar() override;

  virtual NodeOps forwardOps() override { return {}; };
  virtual NodeOps backwardOps() override { return {}; };

  virtual void runForward(const NodeOps& ops) {
    for(auto&& op : ops)
      op();
  }

  virtual void runBackward(const NodeOps& ops) {
    size_t i = 0;
    for(auto&& op : ops)
      if(child(i++)->trainable())
        op();
  }

  virtual void forward() override;

  virtual void backward() override;

  virtual bool trainable() override { return trainable_; }

  virtual void setTrainable(bool trainable) override { trainable_ = trainable; }

  virtual bool memoize() override { return memoize_; };
  virtual void setMemoize(bool memoize) override { memoize_ = memoize; };

  virtual void setId(size_t id) override { id_ = id; }

  virtual size_t getId() override { return id_; }

  virtual void increaseEdges(size_t edges = 1) { edges_ += edges; };
  virtual void decreaseEdges(size_t edges = 1) { edges_ -= edges; };
  virtual size_t edges() { return edges_; };

  virtual Ptr<ExpressionGraph> graph() override { return graph_.lock(); }

  virtual void debug(const std::string& message) override {
    debugMessage_ = message;
    markedForDebug_ = true;
  }

  virtual bool marked_for_debug() override { return markedForDebug_; }
  virtual const std::string& debug_message() override { return debugMessage_; }

  virtual size_t allocate() override;

  virtual void free() override;

  virtual void init() override{};

  virtual void init_dependent() override;

  virtual void set_zero_adjoint() override;

  virtual Tensor& val() override { return val_; };

  virtual Tensor& grad() override { return adj_; };

  virtual const Shape& shape() override { return shape_; }
  virtual const Type& value_type() override { return value_type_; }

  void set_name(const std::string& name) override { name_ = name; }

  const std::string& name() const override { return name_; }

  virtual const std::string form() override { return "box"; }

  virtual const std::string color() override { return "orange"; }

  virtual const std::string label() override {
    std::stringstream label;
    label << "<" << type();
    if(name_ != "none") {
      label << "<br/>"
            << "\"" << name_ << "\"";
    }
    label << " (" << getId() << "/" << trainable() << ")>";
    return label.str();
  }

  virtual std::string graphviz() override {
    std::stringstream ss;
    ss << "\"" << this << "\" [shape=\"" << form() << "\", label=" << label()
       << ", style=\"filled\", fillcolor=\"" << color() << "\"]" << std::endl;
    for(auto&& child : children())
      ss << "\"" << child << "\" -> \"" << this << "\"" << std::endl;
    ss << std::endl;
    return ss.str();
  }

  virtual std::vector<Expr>& children() override { return children_; }

  virtual Expr child(size_t i) override { return children_[i]; }

  Ptr<Backend> getBackend();

  void record(Ptr<AutoTunerRecorder>, size_t, bool) override;
};

struct NaryNodeOp : public Node {
  size_t hash_{0};

  NaryNodeOp(const std::vector<Expr>& nodes,
             Shape shape,
             Type value_type = Type::float32)
      : Node(nodes.front()->graph(), shape, value_type) {
    children_.resize(nodes.size());
    for(size_t i = 0; i < nodes.size(); ++i)
      children_[i] = nodes[i];

    setTrainable(std::any_of(
        nodes.begin(), nodes.end(), [](Expr a) { return a->trainable(); }));

    // Node is to be memoized if all children are to be memoized.
    setMemoize(std::all_of(
        nodes.begin(), nodes.end(), [](Expr a) { return a->memoize(); }));
  }

  NaryNodeOp(const std::vector<Expr>& nodes)
      : NaryNodeOp(nodes, nodes[0]->shape()) {}

  virtual ~NaryNodeOp() {}

  std::vector<Expr>& children() override { return children_; }

  virtual size_t hash() override {
    if(!hash_) {
      std::size_t seed = boost::hash<std::string>()(name());
      boost::hash_combine(seed, type());
      for(size_t i = 0; i < children_.size(); ++i)
        boost::hash_combine(seed, child(i)->hash());
      hash_ = seed;
    }
    return hash_;
  }

  virtual bool equal(Expr node) override {
    if(type() != node->type())
      return false;
    if(name() != node->name())
      return false;
    if(children().size() != node->children().size())
      return false;
    for(size_t i = 0; i < children().size(); ++i)
      if(children()[i]->getId() != node->children()[i]->getId())
        return false;
    return true;
  }
};
}  // namespace marian
