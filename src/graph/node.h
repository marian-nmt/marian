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

  virtual float scalar();

  virtual NodeOps forwardOps() { return {}; };
  virtual NodeOps backwardOps() { return {}; };

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

  virtual void forward();

  virtual void backward();

  virtual bool trainable() { return trainable_; }

  virtual void setTrainable(bool trainable) { trainable_ = trainable; }

  virtual bool memoize() { return memoize_; };
  virtual void setMemoize(bool memoize) { memoize_ = memoize; };

  virtual void setId(size_t id) { id_ = id; }

  virtual size_t getId() { return id_; }

  virtual void increaseEdges(size_t edges = 1) { edges_ += edges; };
  virtual void decreaseEdges(size_t edges = 1) { edges_ -= edges; };
  virtual size_t edges() { return edges_; };

  virtual Ptr<ExpressionGraph> graph() { return graph_.lock(); }

  virtual void debug(const std::string& message) {
    debugMessage_ = message;
    markedForDebug_ = true;
  }

  virtual bool marked_for_debug() { return markedForDebug_; }
  virtual const std::string& debug_message() { return debugMessage_; }

  virtual size_t allocate();

  virtual void free();

  virtual void init(){};

  virtual void init_dependent();

  virtual void set_zero_adjoint();

  virtual Tensor& val() { return val_; };

  virtual Tensor& grad() { return adj_; };

  virtual const Shape& shape() { return shape_; }
  virtual const Type& value_type() { return value_type_; }

  void set_name(const std::string& name) { name_ = name; }

  const std::string& name() const { return name_; }

  virtual const std::string form() { return "box"; }

  virtual const std::string color() { return "orange"; }

  virtual const std::string label() {
    std::stringstream label;
    label << "<" << type();
    if(name_ != "none") {
      label << "<br/>"
            << "\"" << name_ << "\"";
    }
    label << " (" << getId() << "/" << trainable() << ")>";
    return label.str();
  }

  virtual std::string graphviz() {
    std::stringstream ss;
    ss << "\"" << this << "\" [shape=\"" << form() << "\", label=" << label()
       << ", style=\"filled\", fillcolor=\"" << color() << "\"]" << std::endl;
    for(auto&& child : children())
      ss << "\"" << child << "\" -> \"" << this << "\"" << std::endl;
    ss << std::endl;
    return ss.str();
  }

  virtual std::vector<Expr>& children() { return children_; }

  virtual Expr child(size_t i) { return children_[i]; }

  Ptr<Backend> getBackend();

  void record(Ptr<AutoTunerRecorder>, size_t, bool);
};

struct NaryNodeOp : public Node {
  size_t hash_{0};

  NaryNodeOp(const std::vector<Expr>& nodes, Shape shape, Type value_type = Type::float32)
      : Node(nodes.front()->graph(), shape, value_type) {
    children_.resize(nodes.size());
    for(int i = 0; i < nodes.size(); ++i)
      children_[i] = nodes[i];

    setTrainable(std::any_of(
        nodes.begin(), nodes.end(), [](Expr a) { return a->trainable(); }));

    // Node is to be memoized if all children are to be memoized.
    setMemoize(std::all_of(
        nodes.begin(), nodes.end(), [](Expr a) { return a->memoize(); }));

    remove_children_from_top_nodes();
  }

  NaryNodeOp(const std::vector<Expr>& nodes)
      : NaryNodeOp(nodes, nodes[0]->shape()) {}

  virtual ~NaryNodeOp() {}

  std::vector<Expr>& children() { return children_; }

  virtual size_t hash() {
    if(!hash_) {
      std::size_t seed = boost::hash<std::string>()(name());
      boost::hash_combine(seed, type());
      for(int i = 0; i < children_.size(); ++i)
        boost::hash_combine(seed, child(i)->hash());
      hash_ = seed;
    }
    return hash_;
  }

  virtual bool equal(Expr node) {
    if(type() != node->type())
      return false;
    if(name() != node->name())
      return false;
    if(children().size() != node->children().size())
      return false;
    for(int i = 0; i < children().size(); ++i)
      if(children()[i]->getId() != node->children()[i]->getId())
        return false;
    return true;
  }

  void remove_children_from_top_nodes();
};
}
