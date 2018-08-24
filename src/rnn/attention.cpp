#include "attention.h"

#include "graph/node_operators_binary.h"
#include "tensors/tensor_operators.h"

namespace marian {

namespace rnn {

struct AttentionNodeOp : public NaryNodeOp {
  AttentionNodeOp(const std::vector<Expr>& nodes)
      : NaryNodeOp(nodes, newShape(nodes)) {}

  Shape newShape(const std::vector<Expr>& nodes) {
    Shape shape = Shape::broadcast({nodes[1], nodes[2]});

    Shape vaShape = nodes[0]->shape();
    ABORT_IF(vaShape[-2] != shape[-1] || vaShape[-1] != 1, "Wrong size");

    shape.set(-1, 1);
    return shape;
  }

  NodeOps forwardOps() override {
    return {
        NodeOp(Att(val_, child(0)->val(), child(1)->val(), child(2)->val()))};
  }

  NodeOps backwardOps() override {
    return {
      NodeOp(AttBack(child(0)->grad(),
                     child(1)->grad(),
                     child(2)->grad(),
                     child(0)->val(),
                     child(1)->val(),
                     child(2)->val(),
                     adj_);)
    };
  }

  // do not check if node is trainable
  virtual void runBackward(const NodeOps& ops) override {
    for(auto&& op : ops)
      op();
  }

  const std::string type() override { return "Att-ops"; }

  const std::string color() override { return "yellow"; }
};

Expr attOps(Expr va, Expr context, Expr state) {
  std::vector<Expr> nodes{va, context, state};

  int dimBatch = context->shape()[-2];
  int dimWords = context->shape()[-3];
  int dimBeam = 1;
  if(state->shape().size() > 3)
    dimBeam = state->shape()[-4];

  return reshape(Expression<AttentionNodeOp>(nodes),
                 {dimBeam, 1, dimWords, dimBatch});
}
}  // namespace rnn
}  // namespace marian
