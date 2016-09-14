#pragma once

#include "graph.h"
#include "graph_operators.h"
#include "expressions.h"

namespace marian {

template <typename ...Args>
inline Expr input(Args ...args) {
  return Expr(new InputNode(args...));
}

template <typename ...Args>
inline Expr param(Args ...args) {
  return Expr(new ParamNode(args...));
}
template <typename ...Args>
inline Expr constant(Args ...args) {
  return Expr(new ConstantNode(args...));
}

template <typename ...Args>
inline Expr ones(Args ...args) {
  return Expr(new ConstantNode(keywords::value=1, args...));
}

template <typename ...Args>
inline Expr zeroes(Args ...args) {
  return Expr(new ConstantNode(keywords::value=0, args...));
}

/*********************************************************/

inline Expr sigmoid(Expr a) {
  return Expr(new SigmoidNodeOp(a));
}

inline Expr tanh(Expr a) {
  return Expr(new TanhNodeOp(a));
}

inline Expr log(Expr a) {
  return Expr(new LogNodeOp(a));
};

inline Expr exp(Expr a) {
  return Expr(new ExpNodeOp(a));
};

inline Expr operator-(Expr a) {
  return Expr(new NegNodeOp(a));
};

/*********************************************************/

inline Expr operator+(Expr a, Expr b) {
  return Expr(new PlusNodeOp(a, b));
}

inline Expr operator-(Expr a, Expr b) {
  return Expr(new MinusNodeOp(a, b));
}

inline Expr operator*(Expr a, Expr b) {
  return Expr(new MultNodeOp(a, b));
}

inline Expr operator/(Expr a, Expr b) {
  return Expr(new DivNodeOp(a, b));
}

inline Expr dot(Expr a, Expr b) {
  return Expr(new DotNodeOp(a, b));
}

/******************************************************/

Expr broadcast(Shape bShape, Expr a) {
  const Shape& aShape = a.node()->shape();
  if(aShape == bShape) {
    return a;
  }
  else {
    size_t dimsA = aShape.size();
    size_t dimsB = bShape.size();
    UTIL_THROW_IF2(dimsA != dimsB,
                   "Tensor and shape have different number of dimensions");
    for(size_t i = 0; i < dimsA; ++i) {
      int dimA = aShape[i];
      int dimB = bShape[i];
      bool broadcastable = (dimA == dimB || dimA == 1);
      UTIL_THROW_IF2(!broadcastable,
                     "Cannot broadcast tensor dimension "
                     << dimA << " to " << dimB);
      if(dimA == 1 && dimB != 1) {
        std::cerr << "Broadcasting dim " << i << " from " << dimA << " to " << dimB << std::endl;
        if(i == 0) {
          Expr one = ones(keywords::shape={bShape[0], 1});
          a = dot(one, a);
        }
        else if(i == 1) {
          Expr one = ones(keywords::shape={1, bShape[1]});
          a = dot(a, one);
        }
        else {
          UTIL_THROW2("Not implemented");        
        }
      }
    }
    return a;
  }
}

/*********************************************************/

// inefficient
template <typename ...Args>
inline Expr sum(Expr a, Args ...args) {
  using namespace keywords;
  Keywords params(args...);
  int ax = params.Get<int>(axis, whatevs);
  
  ChainPtr n = a.node();
  if(ax == 0) {
    auto lshape = [n]() -> Shape {
      int rows = n->val().shape()[0];
      return {1, rows};
    };
    Expr one = ones(shape={1, n->shape()[0]},
                    lazy_shape=lshape);
    return dot(one, a);        
  }
  else if(ax == 1) {
    auto lshape = [n]() -> Shape {
      int cols = n->val().shape()[1]; 
      //std::cerr << "Shape will be " << cols << " by 1." << std::endl;
      return {cols, 1};
    };
    Expr one = ones(shape={n->shape()[1], 1},
                    lazy_shape=lshape);
    return dot(a, one);          
  }
  else if(ax == 2) {
    UTIL_THROW2("Not implemented");
  }
  else if(ax == 3) {
    UTIL_THROW2("Not implemented");
  }
  return sum(sum(a, axis=0), axis=1);
}

// inefficient
template <typename ...Args>
inline Expr softmax(Expr a, Args ...args) {
  Expr e = exp(a);
#if 0
  ChainPtr n = a.node();
  auto print_shape = [n]() -> Shape {
    std::cerr << "Shape: ";
    for (auto val : n->val().shape()) {
      std::cerr << val << " ";
    }
    std::cerr << std::endl;
    return {1,1};
  };
  using namespace keywords;
  Expr one = ones(shape={1, 1}, lazy_shape=print_shape);
#endif
  
  return e / sum(e, args...);
}

template <typename ...Args>
inline Expr softmax_fast(Expr a, Args ...args) {
  Expr e = Expr(new SoftmaxNodeOp(a, args...));
  return e;
}


// inefficient
template <typename ...Args>
inline Expr mean(Expr a, Args ...args) {
  using namespace keywords;
  Keywords params(args...);
  size_t ax = params.Get<int>(axis, whatevs);

  ChainPtr n = a.node();
  switch (ax) {
    case 0:
      return sum(a, axis=0) / constant(shape={1, 1},
                                       lazy_value=[n]() -> Float {
                                         return n->val().shape()[0];
                                       });
    case 1:
      return sum(a, axis=1) / constant(shape={1, 1},
                                       lazy_value=[n]() -> Float {
                                         return n->val().shape()[1];
                                       });
    case 2:
      UTIL_THROW2("Not implemented");
    case 3:
      UTIL_THROW2("Not implemented");
    default:
      return sum(a) / constant(shape={1, 1},
                               lazy_value=[n]() -> Float {
                                 return n->val().size();
                               });
  }
}

}
