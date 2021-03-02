#pragma once

#include <iostream>
#include <memory>
#include <thread>

#include "common/hash.h"
#include "tensors/backend.h"
#include "tensors/tensor.h"

#include "graph/chainable.h"

namespace marian {

/**
 * Main node class for computation graph,
 * implements most common functions demanded by Chainable.
 * Each operation in a computation graph is a node.
 */
class Node : public Chainable<Tensor> {
protected:
  size_t id_{0};
  size_t edges_{0};
  bool trainable_{true};
  bool destroy_{true};
  bool memoize_{false};

  std::vector<Expr> children_;

  Weak<ExpressionGraph> graph_;
  Shape shape_{1, 1, 1, 1};         // defines the dimensionality of the node (for tensors)
  Type valueType_{Type::float32};   // defines the element type of the node (for tensors)

  std::string name_{"none"};

  Tensor val_{nullptr};  // the resulting new tensor in forward pass
  Tensor adj_{nullptr};  // the accumulated gradients (a tensor) in backward pass

  bool markedForDebug_{false};
  std::string debugMessage_;

  Ptr<std::list<Expr>> subtape_; // a subtape is used to keep track of nodes that need to be freed and recomputed with gradient-checkpointing.
  bool isCheckpoint_{false};     // true if this node has been selected to be a checkpoint, currently only done manually.

  Ptr<AutoTunerRecorder> recorder_;
  size_t recorderHash_;
  bool recorderStop_;

public:
  Node(Ptr<ExpressionGraph> graph, const Shape& shape, const Type& valueType = Type::float32)
    : graph_(graph), shape_(shape), valueType_(valueType) {}

  virtual ~Node() {
    free();
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

  virtual void allocate() override;

  virtual void free() override;

  virtual void init() override {};
  /**
   * Initialization for backward step of top node
   * in computation graph. Allocates memory and sets gradient
   * to 1 (df/df == 1).
   */
  virtual void init_dependent() override;

  /**
   * Initialization for backward step of any non-top node
   * in computation graph. Allocates memory and sets gradient
   * to 0 for further accumulation of gradients from all
   * parents.
   */
  virtual void set_zero_adjoint() override;

  virtual Tensor& val() override { return val_; };

  virtual Tensor& grad() override { return adj_; };

  virtual const Shape& shape() override { return shape_; }
  virtual const Type& value_type() override { return valueType_; }

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
    ss << "\"" << this << "\" ["
      << "shape=\"" << form() << "\", "
      << "label="   << label() << ", "
      << "style=\"filled\", "
      << (isCheckpoint_ ? "penwidth=3, " : "penwidth=1, ")
      << "fillcolor=\"" << color() << "\"];" << std::endl;

    for(auto&& child : children())
      ss << "\"" << child << "\" -> \"" << this << "\";" << std::endl;

    if(subtape_) {
      for(auto&& dep : *subtape_)
        ss << "\"" << dep << "\" -> \"" << this << "\" [style=dotted];" << std::endl;
    }

    ss << std::endl;
    return ss.str();
  }

  virtual std::vector<Expr>& children() override { return children_; }

  virtual Expr child(size_t i) override { return children_[i]; }

  Ptr<Backend> getBackend();

  void record(Ptr<AutoTunerRecorder>, size_t, bool) override;

  // this is currently only called manually by checkpoint(Expr). In the future we will figure out a general algorithm
  virtual void markCheckpoint() override {
    isCheckpoint_ = true;
  }

  virtual bool isCheckpoint() const override {
    return (children_.empty() || isCheckpoint_); // this node is a checkPoint if it's a leaf or if it has been marked.
  }

  virtual void setSubtape(Ptr<std::list<Expr>> subtape) override {
    subtape_ = subtape;
  }

  virtual Ptr<std::list<Expr>> getSubtape() override {
    return subtape_;
  };
};

struct NaryNodeOp : public Node {
  size_t hash_{0};

  // Deduce type automatically, but then all types must be the same
  // this is called automatically when no output type is specified.
  // If the input types are mixed, the output type needs to be specified
  // in the constructor.
  static Type commonType(const std::vector<Expr>& nodes) {
    ABORT_IF(nodes.size() == 0, "NaryNodeOp has no children");
    Type type = nodes[0]->value_type();
    for(int i = 1; i < nodes.size(); ++i)
      ABORT_IF(nodes[i]->value_type() != type,
               "Child {} has different type (first: {} != child: {})",
               i, type, nodes[i]->value_type());
    return type;
  }

  NaryNodeOp(const std::vector<Expr>& nodes)
  : NaryNodeOp(nodes, nodes[0]->shape()) {}

  // this contructor will try to deduce the node type automatically
  NaryNodeOp(const std::vector<Expr>& nodes, Shape shape)
  : NaryNodeOp(nodes, shape, commonType(nodes)) {}

  // this contructor will takes a node type
  NaryNodeOp(const std::vector<Expr>& nodes,
             Shape shape,
             Type value_type)
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

  virtual ~NaryNodeOp() {}

  std::vector<Expr>& children() override { return children_; }

  virtual size_t hash() override {
    if(!hash_) {
      std::size_t seed = util::hash<std::string>()(name());
      util::hash_combine(seed, type());
      util::hash_combine(seed, (size_t)value_type());
      for(size_t i = 0; i < children_.size(); ++i)
        util::hash_combine(seed, child(i)->hash());
      hash_ = seed;
    }
    return hash_;
  }

  virtual bool equal(Expr node) override {
    if(type() != node->type())
      return false;
    else if(name() != node->name())
      return false;
    else if(value_type() != node->value_type())
      return false;
    else if(children().size() != node->children().size())
      return false;
    else {
      for(size_t i = 0; i < children().size(); ++i)
        if(children()[i]->getId() != node->children()[i]->getId())
          return false;
      return true;
    }
  }
};
}  // namespace marian
