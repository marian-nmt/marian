#pragma once

#include "gpu/defs.h"
#include "gpu/placeholders.h"

namespace marian {
  namespace functional {

    template <typename Function, typename X>
    struct UnaryFunctor {
      X x;

      template <class Arg>
      __HD__ UnaryFunctor(Arg a) : x(a) {}

      template <typename ...Args>
      __HDI__ float operator()(Args&&... args) {
        return Function::apply(x(args...));
      }
    };

    template <class Function, class X, class Y>
    struct BinaryFunctor {
      X x;
      Y y;

      template <class Arg1, class Arg2>
      __HD__ BinaryFunctor(Arg1 arg1, Arg2 arg2) : x(arg1), y(arg2) {}

      template <typename ...Args>
      __HDI__ float operator()(Args&&... args) {
        return Function::apply(x(args...), y(args...));
      }
    };

    #define UNARY(name, name2, func) \
    namespace elem { \
      struct name { \
        __HDI__ static float apply(float x) { return func; } \
      }; \
    }\
    template <class X> using name = UnaryFunctor<elem::name, X>;\
    template <typename X>\
    __HDI__ Op<name<X>> name2(Op<X> x) {\
      return Op<name<X>>(name<X>(x.f));\
    }

    #define BINARY(name, name2, func) \
    namespace elem { \
      struct name { \
        __HDI__ static float apply(float x, float y) { return func; } \
      }; \
    }\
    template <class X, class Y> using name = BinaryFunctor<elem::name, X, Y>;\
    template <typename X, typename Y>\
    __HDI__ Op<name<X, Y>> name2(Op<X> x, Op<Y> y) {\
      return Op<name<X, Y>>(name<X, Y>(x.f, y.f));\
    }\
    template <typename X>\
    __HDI__ Op<name<X, C>> name2(Op<X> x, float y) {\
      return name2(x, Op<C>(y));\
    }\
    template <typename Y>\
    __HDI__ Op<name<C, Y>> name2(float x, Op<Y> y) {\
      return name2(Op<C>(x), y);\
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
    UNARY(Logit, logit, x > 0 ? (1.f / (1.f + expf(-x))) : (expf(x) / (1.f + expf(x))));

    BINARY(Plus, operator+, x + y);
    BINARY(Minus, operator-, x - y);
    BINARY(Mult, operator*, x * y);
    BINARY(Div, operator/, x / y);

    #define COMPOUND(name, op) \
    template <typename X, typename Y> \
    __HDI__ auto name(Op<X> x, Op<Y> y) -> decltype(x = x op y) { \
      return x = x op y; \
    }\
    template <typename X> \
    __HDI__ auto name(Op<X> x, float y) -> decltype(x = x op y) { \
      return x = x op y; \
    }

    COMPOUND(operator+=, +);
    COMPOUND(operator-=, -);
    COMPOUND(operator*=, *);
    COMPOUND(operator/=, /);

    UNARY(Negate, operator!, !x);
    BINARY(Eq, operator==, x == y);
    BINARY(NEq, operator!=, x != y);
    BINARY(Gt, operator>, x > y);
    BINARY(Lt, operator<, x < y);
    BINARY(Geq, operator>=, x >= y);
    BINARY(Leq, operator<=, x <= y);
    BINARY(And, operator&&, x && y);
    BINARY(Or, operator||, x || y);

    template <typename T>
    __HDI__ T sgn(T val) {
      return (float(0) < val) - (val < float(0));
    }

    UNARY(Sgn, sgn, sgn(x));

    BINARY(Pow, pow, pow(x, y));

    BINARY(Clip, clip, fabs(x) >= y ? sgn(x) * y : x);

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
      __HD__ TernaryFunctor(Arg1 arg1, Arg2 arg2, Arg3 arg3)
      : x(arg1), y(arg2), z(arg3) {}

      template <typename ...Args>
      __HDI__ float operator()(Args&&... args) {
        return Function::apply(x(args...), y(args...), z(args...));
      }
    };

    #define TERNARY(name, name2, func) \
    namespace elem { \
      struct name { \
        __HDI__ static float apply(float x, float y, float z) { return func; } \
      }; \
    }\
    template <class X, class Y, class Z> using name = TernaryFunctor<elem::name, X, Y, Z>;\
    template <typename X, typename Y, typename Z>\
    __HDI__ Op<name<X, Y, Z>> name2(Op<X> x, Op<Y> y, Op<Z> z) {\
      return Op<name<X, Y, Z>>(name<X, Y, Z>(x.f, y.f, z.f));\
    }\
    template <typename X, typename Z>\
    __HDI__ Op<name<X, C, Z>> name2(Op<X> x, float y, Op<Z> z) {\
      return name2(x, Op<C>(y), z);\
    }\
    template <typename Y, typename Z>\
    __HDI__ Op<name<C, Y, Z>> name2(float x, Op<Y> y, Op<Z> z) {\
      return name2(Op<C>(x), y, z);\
    }\
    template <typename X>\
    __HDI__ Op<name<X, C, C>> name2(Op<X> x, float y, float z) {\
      return name2(x, Op<C>(y), Op<C>(z));\
    }\
    template <typename Y>\
    __HDI__ Op<name<C, Y, C>> name2(float x, Op<Y> y, float z) {\
      return name2(Op<C>(x), y, Op<C>(z));\
    }\
    template <typename Z>\
    __HDI__ Op<name<C, C, Z>> name2(float x, float y, Op<Z> z) {\
      return name2(Op<C>(x), Op<C>(y), z);\
    }

    TERNARY(IfThenElse, if_then_else, x ? y : z);

  }
}