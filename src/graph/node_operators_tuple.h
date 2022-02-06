#pragma once

#include "graph/node_operators_unary.h"

namespace marian {

// Base class for a node that has more than one forward value tensor. 
// For now we only do one additional value.
class TupleNode {
protected:
  Tensor tupleVal_; // the additional forward value tensor

  friend struct TupleViewNodeOp;

  // Force implementation of these functions. The node that inherits from this 
  // should use them during allocation/deallocation.
  virtual void allocateTuple() = 0;
  virtual void freeTuple() = 0;

public:
  // The specific node needs to implement what the view is actually looking at
  virtual Expr tupleView() = 0;
};

// This is a view similar to ReshapeNodeOp or SliceViewNodeOp
// that uses the additional value as its tensor without 
// allocating or destroying anthing. This has the purpose to 
// create an Expression that can be put on the tape of the graph and
// visited in correct topological order. It should also make sure that
// the node that actually holds the memory persists via the reference
// to tupleNode_. 
struct TupleViewNodeOp : public UnaryNodeOp {
private:
  Expr tupleNode_; // hold a reference to the actually node with the tuple data

public:
  TupleViewNodeOp(Expr origin, Shape shape, Type type)
  : UnaryNodeOp(origin, shape, type),
    tupleNode_(origin) {
    Node::destroy_ = false;   // should not be detroyed or freed, the origin node is handling that
    Node::trainable_ = false; // for now this is not trainable
  }

  // make sure these functions don't actually do anything, origin node handles all this
  ~TupleViewNodeOp() {}
  void allocate() override {}
  void free() override {}
  void forward() override {}
  void backward() override {}
  void init_dependent() override {}
  void set_zero_adjoint() override {}

  // Return the additional tuple tensor as the val() tensor for the view
  Tensor& val() override {
    // we have to use a raw pointer cast here as we cannot add IntrusivePtr ref-counting to TupleNode
    // otherwise we would have ambigous inheritence
    auto tptr = dynamic_cast<TupleNode*>(tupleNode_.get());
    ABORT_IF(!tptr, "Could not convert to tuple?");
    return tptr->tupleVal_;
  };

  // Currently not trainable. We will see if that will be useful at some point
  Tensor& grad() override {
    ABORT("There should be no gradients for tuple values");
  };

  const std::string type() override { return "tupleView"; }
  const std::string color() override { return "grey"; }
};

// This is an implementation of topk, similar to the PyTorch node.
// At the moment we only handle axis=-1 in here, but do transposes
// in the actual operator to handle other axes (inefficiently).
// The normal forward values here are the top-k values per axis,
// the additional value from the TupleNode contains the integer
// indices of the top-k values.
struct TopKNodeOp : public UnaryNodeOp, 
                    public TupleNode { 
private:
  int k_;           // how many top-k results?
  int axis_;        // on which axis
  bool descending_; // sort-order, by default descending. PyTorch has a version without sorting, we always sort.

public:
  TopKNodeOp(Expr a, int k, int axis, bool descending = true) 
  : UnaryNodeOp(a, newShape(a, k, axis)), 
    k_{k}, descending_{descending} {}

  Shape newShape(Expr a, int k, int axis) {
    Shape shape = a->shape();
    axis_ = shape.axis(axis);

    shape.set(axis_, k);
    return shape;
  }

  // imlementation of TupleNode-specific pure-virtual functions for allocation
  void allocateTuple() override final {
    graph()->getTensorAllocator()->allocate(tupleVal_, shape(), Type::uint32);
  }

  // we override the normal allocation to include the TupleNode allocation
  void allocate() override {
    UnaryNodeOp::allocate();
    allocateTuple();
  }

  // imlementation of TupleNode-specific pure-virtual functions for de-allocation
  void freeTuple() override final {
    if(graph()) {
      if(tupleVal_) {
        graph()->free(tupleVal_);
        tupleVal_ = nullptr;
      }
    }
  }

  // we override the normal allocation to include the TupleNode de-allocation
  void free() override {
    UnaryNodeOp::free();
    freeTuple();
  }

  // Create and return a TupleView to the additional forward value
  virtual Expr tupleView() override final {
    return Expression<TupleViewNodeOp>(this, shape(), Type::uint32);
  }

  void forward() override {
    TopK(/*out*/val_, /*out: topkIndices=*/tupleVal_,
         graph()->allocator(),
         child(0)->val(), k_, axis_, descending_);
  }

  void backward() override {
    Insert</*add=*/true>(/*out*/child(0)->grad(), adj_, val_, axis_);
  }

  const std::string type() override { return "topk"; }

  virtual size_t hash() override {
    if(!hash_) {
      hash_ = NaryNodeOp::hash();
      util::hash_combine(hash_, k_);
      util::hash_combine(hash_, axis_);
      util::hash_combine(hash_, descending_);
    }
    return hash_;
  }

  virtual bool equal(Expr node) override {
    if(!NaryNodeOp::equal(node))
      return false;
    auto cnode = std::dynamic_pointer_cast<TopKNodeOp>(node);
    if(!cnode)
      return false;
    if(k_ != cnode->k_)
      return false;    
    if(axis_ != cnode->axis_)
      return false;
    if(descending_ != cnode->descending_)
      return false;
    return true;
  }
};

}
