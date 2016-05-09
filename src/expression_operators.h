#pragma once

#include "graph.h"
#include "graph_operators.h"
#include "expressions.h"

namespace marian {

template <typename ...Args>
inline Expr data(Args ...args) {
  return Expr(new DataNode(args...));
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

Expr broadcast(Shape shape, Expr a) {
  if(a.val().shape() == shape) {
    return a;
  }
  else {
    size_t dimsA = a.val().shape().size();
    size_t dimsB = shape.size();
    UTIL_THROW_IF2(dimsA != dimsB,
                   "Tensor and shape have different number of dimensions");
    for(size_t i = 0; i < dimsA; ++i) {
      int dimA = a.val().shape()[i];
      int dimB = shape[i];
      bool broadcastable = (dimA == dimB || dimA == 1);
      UTIL_THROW_IF2(!broadcastable,
                     "Cannot broadcast tensor dimension "
                     << dimA << " to " << dimB);
      if(dimA == 1 && dimB > 1) {
        std::cerr << "Broadcasting dim " << i << " from " << dimA << " to " << dimB << std::endl;
        if(i == 0) {
          Expr one = ones(keywords::shape={shape[0], 1});
          a = dot(one, a);
        }
        else if(i == 1) {
          Expr one = ones(keywords::shape={1, shape[1]});
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
  
  if(ax == 0) {
    auto lshape = [&a]() -> Shape {
      int rows = a.val().shape()[0];
      return {1, rows};
    };
    Expr one = ones(lazy_shape=lshape);
    return dot(one, a);        
  }
  else if(ax == 1) {
    auto lshape = [&a]() -> Shape {
      int cols = a.val().shape()[1]; 
      return {cols, 1};
    };
    Expr one = ones(lazy_shape=lshape);
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
  return e / sum(e, args...);
}

// inefficient
template <typename ...Args>
inline Expr mean(Expr a, Args ...args) {
  using namespace keywords;
  Keywords params(args...);
  size_t ax = params.Get<int>(axis, whatevs);

  switch (ax) {
    case 0:
      return sum(a, axis=0) / constant(shape={1, 1},
                                       lazy_value=[&a]() -> Float {
                                         return a.val().shape()[0];
                                       });
    case 1:
      return sum(a, axis=1) / constant(shape={1, 1},
                                       lazy_value=[&a]() -> Float {
                                         return a.val().shape()[1];
                                       });
    case 2:
      UTIL_THROW2("Not implemented");
    case 3:
      UTIL_THROW2("Not implemented");
    default:
      return sum(a) / constant(shape={1, 1},
                               lazy_value=[&a]() -> Float {
                                 return a.val().size();
                               });
  }
}

}