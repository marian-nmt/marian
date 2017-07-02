#include "rnn/attention.h"

#include "graph/node_operators_binary.h"
#include "kernels/tensor_operators.h"

namespace marian {

namespace rnn {

struct AttentionNodeOp : public NaryNodeOp {
  AttentionNodeOp(const std::vector<Expr>& nodes)
      : NaryNodeOp(nodes, keywords::shape = newShape(nodes)) {}

  Shape newShape(const std::vector<Expr>& nodes) {
    Shape shape = nodes[1]->shape();

    Shape vaShape = nodes[0]->shape();
    Shape ctxShape = nodes[1]->shape();
    Shape stateShape = nodes[2]->shape();

    for(int i = 0; i < stateShape.size(); ++i) {
      UTIL_THROW_IF2(ctxShape[i] != stateShape[i] && ctxShape[i] != 1
                         && stateShape[i] != 1,
                     "Shapes cannot be broadcasted");
      shape.set(i, std::max(ctxShape[i], stateShape[i]));
    }

    UTIL_THROW_IF2(vaShape[0] != shape[1] || vaShape[1] != 1, "Wrong size");

    shape.set(1, 1);
    return shape;
  }

  NodeOps forwardOps() {
    return {NodeOp(Att(val_,
                       child(0)->val(),
                       child(1)->val(),
                       child(2)->val(),
                       children_.size() == 4 ? child(3)->val() : nullptr))};
  }

  NodeOps backwardOps() {
    return {
      NodeOp(AttBack(child(0)->grad(),
                     child(1)->grad(),
                     child(2)->grad(),
                     children_.size() == 4 ? child(3)->grad() : nullptr,
                     child(0)->val(),
                     child(1)->val(),
                     child(2)->val(),
                     children_.size() == 4 ? child(3)->val() : nullptr,
                     adj_);)
    };
  }

  // do not check if node is trainable
  virtual void runBackward(const NodeOps& ops) {
    for(auto&& op : ops)
      op();
  }

  const std::string type() { return "Att-ops"; }

  const std::string color() { return "yellow"; }
};

Expr attOps(Expr va, Expr context, Expr state, Expr coverage) {
  std::vector<Expr> nodes{va, context, state};
  if(coverage)
    nodes.push_back(coverage);

  int dimBatch = context->shape()[0];
  int dimWords = context->shape()[2];
  int dimBeam = state->shape()[3];
  return reshape(Expression<AttentionNodeOp>(nodes),
                 {dimWords, dimBatch, 1, dimBeam});
}

}
}