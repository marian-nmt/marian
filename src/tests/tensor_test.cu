#include <iostream>
#include <type_traits>
#include <typeinfo>

#include "functional/defs.h"
#include "functional/functional.h"
#include "functional/floats.h"

  using namespace marian::functional;

//template <int N>
//struct Select {
//  template <typename T, typename ...Args>
//  __HDI__ static auto apply(T&& arg, Args&&... args) -> decltype(Select<N-1>::apply(args...)) {
//    return Select<N-1>::apply(args...);
//  }
//};
//
//template <>
//struct Select<0> {
//  template <typename T, typename ...Args>
//  __HDI__ static T apply(T&& arg, Args&&... args) {
//    return arg;
//  }
//};
//
//template <int N>
//struct Var;
//
//template <int V>
//struct C {
//  template <typename ...Args>
//  __HDI__ float operator()(Args&&... args) { return V; }
//};
//

//
///******************************************************************************/
//
//template <int N>
//struct Var {
//
//  template <typename ...Args>
//  __HDI__ float& operator()(Args&&... args) {
//    return Select<N-1>::apply(args...);
//  }
//};


//
///******************************************************************************/
//
//struct Capture {
//  float value;
//
//  __HD__ Capture(float val) : value(val) {};
//
//  template <typename ...Args>
//  __HDI__ float operator()(Args&&... args) { return value; }
//};
//
//__HDI__ Capture c(float x) {
//  return Capture(x);
//}



///******************************************************************************/
//
//template <class X, class Y>
//struct Minus {
//  X x;
//  Y y;
//
//  __HD__ Minus(X _x, Y _y) : x(_x), y(_y) {}
//
//  template <typename ...Args>
//  __HDI__ float operator()(Args&&... args) {
//    return x(args...) - y(args...);
//  }
//};
//
///******************************************************************************/
//
//template <class X, class Y>
//struct Plus {
//  X x;
//  Y y;
//
//  __HD__ Plus(X _x, Y _y) : x(_x), y(_y) {}
//
//  template <typename ...Args>
//  __HDI__ float operator()(Args&&... args) {
//    return x(args...) + y(args...);
//  }
//};
//
//template <class C>
//using IsClass = typename std::enable_if<std::is_class<C>::value, C>::type;
//
//template <class X, class Y>
//__HDI__ Plus<IsClass<X>, IsClass<Y>> operator+(X x, Y y) {
//  return Plus<X, Y>(x, y);
//}
//
//template <class Y>
//__HDI__ Plus<Capture, IsClass<Y>> operator+(Capture x, Y y) {
//  return Plus<Capture, Y>(x, y);
//}
//
//template <class X>
//__HDI__ Plus<IsClass<X>, Capture> operator+(X x, Capture y) {
//  return Plus<X, Capture>(x, y);
//}
//
///******************************************************************************/
//
//template <class X, class Y>
//struct Mult {
//  X x;
//  Y y;
//
//  __HD__ Mult(X _x, Y _y) : x(_x), y(_y) {}
//
//  template <typename ...Args>
//  __HDI__ float operator()(Args&&... args) {
//    return x(args...) * y(args...);
//  }
//};
//
//template <class X, class Y>
//__HDI__ Mult<X, Y> operator*(X x, Y y) {
//  return Mult<X, Y>(x, y);
//}
//
///******************************************************************************/
//
//template <class X, class Y>
//struct Div {
//  X x;
//  Y y;
//
//  __HD__ Div(X _x, Y _y) : x(_x), y(_y) {}
//
//  template <typename ...Args>
//  __HDI__ float operator()(Args&&... args) {
//    return x(args...) / y(args...);
//  }
//};
//
//template <class X, class Y>
//__HDI__ Div<X, Y> operator/(X x, Y y) {
//  return Div<X, Y>(x, y);
//}
//
///******************************************************************************/
//
//template <typename X>
//struct Tanh {
//  X x;
//
//  __HD__ Tanh(X _x) : x(_x) {}
//
//  template <typename ...Args>
//  __HDI__ float operator()(Args&&... args) {
//    return tanhf(x(args...));
//  }
//};
//
//template <class X>
//__HDI__ Tanh<X> tanh(X x) {
//  return Tanh<X>(x);
//}
//
///******************************************************************************/
//
//__HDI__ float logitf(float val) {
//  if(val >= 0.f) {
//    float z = expf(-val);
//    return float(1) / (float(1) + z);
//  }
//  else {
//    float z = expf(val);
//    return z / (float(1) + z);
//  }
//}
//
//template <typename X>
//struct Sigmoid {
//  X x;
//
//  __HD__ Sigmoid(X _x) : x(_x) {}
//
//  template <typename ...Args>
//  __HDI__ float operator()(Args&&... args) {
//    return logitf(x(args...));
//  }
//};
//
//template <class X>
//__HDI__ Sigmoid<X> logit(X x) {
//  return Sigmoid<X>(x);
//}
//
///******************************************************************************/
//
//template <typename X>
//struct Exp {
//  X x;
//
//  __HD__ Exp(X _x) : x(_x) {}
//
//  template <typename ...Args>
//  __HDI__ float operator()(Args&&... args) {
//    return expf(x(args...));
//  }
//};
//
//template <class X>
//__HDI__ Exp<X> exp(X x) {
//  return Exp<X>(x);
//}
//
//
///******************************************************************************/

template <int N>
__HDI__ C<N> simple(C<N> c) {
  return c;
}

template <int N>
__HDI__ Var<N> simple(Var<N> v) {
  return v;
}

__HDI__ Capture simple(Capture c) {
  return c;
}

template <class X>
__HDI__ auto cut(Plus<X, C<0>> f)->decltype(f.x) {
  return f.x;
};

template <class Y>
__HDI__ auto cut(Plus<C<0>, Y> f)->decltype(f.y) {
  return f.y;
};

template <class Y>
__HDI__ C<0> cut(Plus<C<0>, C<0>> f) {
  return C<0>();
};

template <class X, class Y>
__HDI__ auto simple(Plus<X, Y> f)->decltype(cut(simple(f.x) + simple(f.y))) {
  return cut(simple(f.x) + simple(f.y));
};

template <class X>
__HDI__ C<0> cut(Mult<X, C<0>> f) {
  return C<0>();
};

template <class Y>
__HDI__ C<0> cut(Mult<C<0>, Y> f) {
  return C<0>();
};

template <class X>
__HDI__ auto cut(Mult<X, C<1>> f)->decltype(f.x) {
  return f.x;
};

template <class Y>
__HDI__ auto cut(Mult<C<1>, Y> f)->decltype(f.y) {
  return f.y;
};

template <class X, class Y>
__HDI__ auto simple(Mult<X, Y> f)->decltype(cut(simple(f.x) * simple(f.y))) {
  return cut(simple(f.x) * simple(f.y));
};


//template <class X>
//__HDI__ C<0> cut(Minus<X, X> f) {
//  return C<0>();
//};
//
//template <class X>
//__HDI__ auto cut(Minus<X, C<0>> f)->decltype(f.x) {
//  return f.x;
//};
//
//template <class X, class Y>
//__HDI__ auto simple(Minus<X, Y> f)->decltype(cut(simple(f.x) - simple(f.y))) {
//  return cut(simple(f.x) - simple(f.y));
//};
//
//template <class X>
//__HDI__ C<0> cut(Mult<X, C<0>> f) {
//  return C<0>();
//};
//
//template <class Y>
//__HDI__ C<0> cut(Mult<C<0>, Y> f) {
//  return C<0>();
//};
//
//template <class X>
//__HDI__ auto cut(Mult<X, C<1>> f)->decltype(f.x) {
//  return f.x;
//};
//
//template <class Y>
//__HDI__ auto cut(Mult<C<1>, Y> f)->decltype(f.y) {
//  return f.y;
//};
//
//template <class X, class Y>
//__HDI__ auto simple(Mult<X, Y> f)->decltype(cut(simple(f.x) * simple(f.y))) {
//  return cut(simple(f.x) * simple(f.y));
//};
//
//__HDI__ C<1> cut(Exp<C<0>>) { return C<1>(); };
//
//template <class X>
//__HDI__ auto simple(Exp<X> f)->decltype(cut(exp(simple(f.x)))) {
//  return cut(exp(simple(f.x)));
//};
//
//template <class X>
//__HDI__ auto cut(Div<X, C<1>> f)->decltype(f.x) {
//  return f.x;
//};
//
//template <class Y>
//__HDI__ C<0> cut(Div<C<0>, Y> f) {
//  return C<0>();
//};
//
//template <class X>
//__HDI__ C<1> cut(Div<X, X> f) {
//  return C<1>();
//};
//
//template <class X, class Y>
//__HDI__ auto simple(Div<X, Y> f)->decltype(cut(simple(f.x) / simple(f.y))) {
//  return cut(simple(f.x) / simple(f.y));
//};
//
//template <template <class> class F, class X>
//__HDI__ auto simple(F<X> f)->decltype(F<decltype(simple(f.x))>(simple(f.x))) {
//  return F<decltype(simple(f.x))>(simple(f.x));
//};
//
//template <template <class, class> class F, class X, class Y>
//__HDI__ auto simple(F<X, Y> f)->decltype(F<decltype(simple(f.x)), decltype(simple(f.y))>(simple(f.x), simple(f.y))) {
//  return F<decltype(simple(f.x)), decltype(simple(f.y))>(simple(f.x), simple(f.y));
//};
//
//
///******************************************************************************/
//
//template <class X, class Y>
//__HDI__ Minus<X, Y> operator-(X x, Y y) {
//  return Minus<X, Y>(x, y);
//}
//

//template <typename X, typename Y, int N>
//__HDI__ auto grad(Minus<X, Y> f, Var<N> g)->decltype((grad(f.x, g) - grad(f.y, g))) {
//  return (grad(f.x, g) - grad(f.y, g));
//}
//
//template <typename X, typename Y, int N>
//__HDI__ auto grad(Plus<X, Y> f, Var<N> g)->decltype((grad(f.x, g) + grad(f.y, g))) {
//  return (grad(f.x, g) + grad(f.y, g));
//}


//template <typename X, typename Y, int N>
//__HDI__ auto grad(Div<X, Y> f, Var<N> g)->decltype(((grad(f.x, g) * f.y - f.x * grad(f.y, g)) / (f.y * f.y))) {
//  return ((grad(f.x, g) * f.y - f.x * grad(f.y, g)) / (f.y * f.y));
//}
//
//template <typename X, int N>
//__HDI__ auto grad(Tanh<X> f, Var<N> g)->decltype((C<1>() - f * f) * grad(f.x, g)) {
//  return (C<1>() - f * f) * grad(f.x, g);
//}
//
//template <typename X, int N>
//__HDI__ auto grad(Sigmoid<X> f, Var<N> g)->decltype(f * (C<1>() - f) * grad(f.x, g)) {
//  return f * (C<1>() - f) * grad(f.x, g);
//}

///******************************************************************************/
//
//struct Node {
//  virtual float forward(float) = 0;
//  virtual float backward(float) = 0;
//};
//
//template <class F, class DF>
//struct UnaryNode : public Node {
//  F f;
//  DF df;
//
//  UnaryNode(F f_, DF df_) : Node(), f(f_), df(df_) {}
//
//  float forward(float x) {
//    return f(x);
//  }
//
//  float backward(float x) {
//    return df(x);
//  }
//};
//
//template <class F, class DF>
//Node* unary(F f, DF df) {
//  return new UnaryNode<F, DF>(f, df);
//}
//
//template <class F>
//Node* unary(F f) {
//  Var<1> x;
//  auto df = grad(f, x);
//  return unary(f, df);
//}

template <int N, int K>
C<N == K> grad(ref<N>, ref<K>) {
  return C<N == K>();
}

template <int N>
C<0> grad(Capture, ref<N>) {
  return C<0>();
}

template <int V, int N>
C<0> grad(C<V>, ref<N>) {
  return C<0>();
}


template <typename X, typename Y, int N>
auto grad(Mult<X, Y> f, ref<N> g)->decltype(grad(f.x, g) * f.y + f.x * grad(f.y, g)) {
  return grad(f.x, g) * f.y + f.x * grad(f.y, g);
}

//template <typename X, typename Y, int N>
//auto grad(Plus<X, Y> f, ref<N> g)->decltype(grad(f.x, g) + grad(f.y, g)) {
//  return grad(f.x, g) + grad(f.y, g);
//}

template <typename X, int N>
auto grad(Exp<X> f, ref<N> g)->decltype(cut(f * grad(f.x, g))) {
  return cut(f * grad(f.x, g));
}

int main(int argc, char** argv) {

  ref<1> x;
  ref<2> y;

  std::cerr <<  F<f2i(1.5)>::value << std::endl;
  //auto df = grad(f, x);

  //std::cerr << f(0.f) << std::endl;
  //std::cerr << f.to_string() << std::endl;
  //std::cerr << df.to_string() << std::endl;

  return 0;
}
