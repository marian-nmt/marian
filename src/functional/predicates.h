#pragma once

#include "functional/defs.h"
#include "functional/operands.h"

namespace marian {
namespace functional {

template <typename Function, typename X>
struct UnaryFunctor {
  X x;

  template <class Arg>
  UnaryFunctor(Arg a) : x(a) {}

  template <typename... Args>
  __HDI__ float operator()(Args&&... args) {
    return Function::apply(x(args...));
  }

  std::string to_string() { return Function::n() + "<" + x.to_string() + ">"; }
};

template <class Function, class X, class Y>
struct BinaryFunctor {
  X x;
  Y y;

  template <class Arg1, class Arg2>
  BinaryFunctor(Arg1 arg1, Arg2 arg2) : x(arg1), y(arg2) {}

  template <typename... Args>
  __HDI__ float operator()(Args&&... args) {
    return Function::apply(x(args...), y(args...));
  }

  std::string to_string() {
    return Function::n() + "<" + x.to_string() + "," + y.to_string() + ">";
  }
};

#define UNARY(name, name2, func)                         \
  namespace elem {                                       \
  struct name {                                          \
    __HDI__ static float apply(float x) { return func; } \
    static std::string n() { return #name; }             \
  };                                                     \
  }                                                      \
  template <class X>                                     \
  using name = UnaryFunctor<elem::name, X>;              \
  template <typename X>                                  \
  name<IsClass<X>> name2(X x) {                          \
    return name<X>(x);                                   \
  }                                                      \
  static name<Capture> name2(Capture x) { return name<Capture>(x); }

#define BINARY(name, name2, func)                                 \
  namespace elem {                                                \
  struct name {                                                   \
    __HDI__ static float apply(float x, float y) { return func; } \
    static std::string n() { return #name; }                      \
  };                                                              \
  }                                                               \
  template <class X, class Y>                                     \
  using name = BinaryFunctor<elem::name, X, Y>;                   \
  template <class X, class Y>                                     \
  name<IsClass<X>, IsClass<Y>> name2(X x, Y y) {                  \
    return name<X, Y>(x, y);                                      \
  }                                                               \
  template <class Y>                                              \
  name<Capture, IsClass<Y>> name2(Capture x, Y y) {               \
    return name<Capture, Y>(x, y);                                \
  }                                                               \
  template <class X>                                              \
  name<IsClass<X>, Capture> name2(X x, Capture y) {               \
    return name<X, Capture>(x, y);                                \
  }

UNARY(Tanh, tanh, tanhf(x));
UNARY(Sin, sin, sinf(x));
UNARY(Cos, cos, cosf(x));
UNARY(Tan, tan, tanf(x));
UNARY(Log, log, logf(x));
UNARY(Exp, exp, expf(x));
UNARY(Abs, abs, fabs(x));
UNARY(Sqrt, sqrt, sqrtf(x));
UNARY(Neg, operator-, -x);
UNARY(Logit,
      logit,
      x > 0 ? (1.f / (1.f + expf(-x))) : (expf(x) / (1.f + expf(x))));

BINARY(Plus, operator+, x + y);
BINARY(Minus, operator-, x - y);
BINARY(Mult, operator*, x* y);
BINARY(Div, operator/, x / y);

UNARY(Negate, operator!, !x);
BINARY(Eq, operator==, x == y);
BINARY(NEq, operator!=, x != y);
BINARY(Gt, operator>, x> y);
BINARY(Lt, operator<, x<y);
BINARY(Geq, operator>=, x >= y);
BINARY(Leq, operator<=, x <= y);
BINARY(And, operator&&, x&& y);
BINARY(Or, operator||, x || y);

template <typename T>
__HDI__ T sgn(T val) {
  return (float(0) < val) - (val < float(0));
}

UNARY(Sgn, sgn, sgn(x));

BINARY(Pow, pow, pow(x, y));

BINARY(Clip, clip, fabs(x) >= y ? sgn(x) * y : x);
BINARY(Bump, bump, fabs(x) >= y ? 0.f : 1.f);

UNARY(sReLU, ReLU, x > 0.f ? x : 0.f);
UNARY(sReLUBack, ReLUback, x > 0.f ? 1.f : 0.f);
BINARY(sPReLU, PReLU, x > 0.f ? x : x * y);
BINARY(sPReLUBack, PReLUback, x > 0.f ? 1.f : y);

template <class Function, class X, class Y, class Z>
struct TernaryFunctor {
  X x;
  Y y;
  Z z;

  template <class Arg1, class Arg2, class Arg3>
  TernaryFunctor(Arg1 arg1, Arg2 arg2, Arg3 arg3) : x(arg1), y(arg2), z(arg3) {}

  template <typename... Args>
  __HDI__ float operator()(Args&&... args) {
    return Function::apply(x(args...), y(args...), z(args...));
  }
};

#define TERNARY(name, name2, func)                                         \
  namespace elem {                                                         \
  struct name {                                                            \
    __HDI__ static float apply(float x, float y, float z) { return func; } \
  };                                                                       \
  }                                                                        \
  template <class X, class Y, class Z>                                     \
  using name = TernaryFunctor<elem::name, X, Y, Z>;                        \
  template <typename X, typename Y, typename Z>                            \
  name<IsClass<X>, IsClass<Y>, IsClass<Z>> name2(X x, Y y, Z z) {          \
    return name<X, Y, Z>(x, y, z);                                         \
  }                                                                        \
  template <typename X, typename Z>                                        \
  name<IsClass<X>, Capture, IsClass<Z>> name2(X x, Capture y, Z z) {       \
    return name<X, Capture, Z>(x, y, z);                                   \
  }                                                                        \
  template <typename Y, typename Z>                                        \
  name<Capture, IsClass<Y>, IsClass<Z>> name2(Capture x, Y y, Z z) {       \
    return name<Capture, Y, Z>(x, y, z);                                   \
  }                                                                        \
  template <typename X>                                                    \
  name<IsClass<X>, Capture, Capture> name2(X x, Capture y, Capture z) {    \
    return name<X, Capture, Capture>(x, y, z);                             \
  }                                                                        \
  template <typename Y>                                                    \
  name<Capture, IsClass<Y>, Capture> name2(Capture x, Y y, Capture z) {    \
    return name<Capture, Y, Capture>(x, y, z);                             \
  }                                                                        \
  template <typename Z>                                                    \
  name<Capture, Capture, IsClass<Z>> name2(Capture x, Capture y, Z z) {    \
    return name<Capture, Capture, Z>(x, y, z);                             \
  }

TERNARY(IfThenElse, if_then_else, x ? y : z);

template <class X, class Y>
struct Assign {
  X x;
  Y y;

  template <class Arg1, class Arg2>
  Assign(Arg1 arg1, Arg2 arg2) : x(arg1), y(arg2) {}

  template <typename... Args>
  __HDI__ float operator()(Args&&... args) {
    return x(args...) = y(args...);
  }
};

template <int N>
struct Assignee {
  Var<N> var;

  Assignee() {}
  Assignee(Var<N> v) : var(v) {}

  template <typename... Args>
  __HDI__ float& operator()(Args&&... args) {
    return var(args...);
  }

  template <class X>
  Assign<Var<N>, IsClass<X>> operator=(X x) {
    return Assign<Var<N>, X>(var, x);
  }

  Assign<Var<N>, Capture> operator=(Capture x) {
    return Assign<Var<N>, Capture>(var, x);
  }

  template <class X>
  auto operator+=(X x) -> decltype(*this = *this + x) {
    return *this = *this + x;
  }

  template <class X>
  auto operator-=(X x) -> decltype(*this = *this - x) {
    return *this = *this - x;
  }

  template <class X>
  auto operator*=(X x) -> decltype(*this = *this * x) {
    return *this = *this * x;
  }

  template <class X>
  auto operator/=(X x) -> decltype(*this = *this / x) {
    return *this = *this / x;
  }

  std::string to_string() { return var.to_string(); }
};

/******************************************************************************/
}
}
