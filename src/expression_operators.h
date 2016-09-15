#pragma once

#include "expression_graph.h"

namespace marian {

Expr named(Expr a, const std::string& name);

Expr logit(Expr a);

Expr tanh(Expr a);

Expr log(Expr a);

Expr exp(Expr a);

Expr operator-(Expr a);

/*********************************************************/

Expr operator+(Expr a, Expr b);

Expr operator-(Expr a, Expr b);

Expr operator*(Expr a, Expr b);

Expr operator/(Expr a, Expr b);

Expr dot(Expr a, Expr b);

/******************************************************/

Expr broadcast(Shape bShape, Expr a);

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
    Expr one = a.graph()->ones(shape={1, n->shape()[0]},
                    lazy_shape=lshape);
    return dot(one, a);        
  }
  else if(ax == 1) {
    auto lshape = [n]() -> Shape {
      int cols = n->val().shape()[1]; 
      //std::cerr << "Shape will be " << cols << " by 1." << std::endl;
      return {cols, 1};
    };
    Expr one = a.graph()->ones(shape={n->shape()[1], 1},
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
Expr softmax(Expr a, Args ...args) {
  Expr e = exp(a);
  return e / sum(e, args...);
}

Expr softmax_fast(Expr a);

// inefficient
template <typename ...Args>
inline Expr mean(Expr a, Args ...args) {
  using namespace keywords;
  Keywords params(args...);
  size_t ax = params.Get<int>(axis, whatevs);

  ChainPtr n = a.node();
  switch (ax) {
    case 0:
      return sum(a, axis=0) / a.graph()->constant(shape={1, 1},
                                       lazy_value=[n]() -> Float {
                                         return n->val().shape()[0];
                                       });
    case 1:
      return sum(a, axis=1) / a.graph()->constant(shape={1, 1},
                                       lazy_value=[n]() -> Float {
                                         return n->val().shape()[1];
                                       });
    case 2:
      UTIL_THROW2("Not implemented");
    case 3:
      UTIL_THROW2("Not implemented");
    default:
      return sum(a) / a.graph()->constant(shape={1, 1},
                               lazy_value=[n]() -> Float {
                                 return n->val().size();
                               });
  }
}

}
