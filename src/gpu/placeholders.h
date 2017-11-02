#pragma once

#include "gpu/defs.h"

namespace marian {
  namespace functional {

    template <int N>
    struct Select {
      template <typename T, typename ...Args>
      __HDI__ static auto apply(T&& arg, Args&&... args) -> decltype(Select<N-1>::apply(args...)) {
        return Select<N-1>::apply(args...);
      }
    };

    template <>
    struct Select<0> {
      template <typename T, typename ...Args>
      __HDI__ static T apply(T&& arg, Args&&... args) {
        return arg;
      }
    };

    template <int N>
    struct X {

      template <typename ...Args>
      __HDI__ float& operator()(Args&&... args) {
        return Select<N-1>::apply(args...);
      }
    };

    struct C {
      float value;

      __HD__ C(const C& c) : value(c.value) {}
      __HD__ C(float f) : value(f) {}

      template <typename ...Args>
      __HDI__ float& operator()(Args&&... args) { return value; }
    };


    template <class X, class Y>
    struct Assign {
      X x;
      Y y;

      template <class Arg1, class Arg2>
      __HD__ Assign(Arg1&& arg1, Arg2&& arg2) : x(arg1), y(arg2) {}

      template <typename ...Args>
      __HDI__ float operator()(Args&&... args) {
        return x(args...) = y(args...);
      }
    };

    template <class F>
    struct Op {
      F f;

      __HD__ Op() {}

      template <class A>
      __HD__ Op(A a) : f(a) {}

      template <class X>
      __HD__ Op<Assign<F, X>> operator=(Op<X> x) {
        return Op<Assign<F, X>>(Assign<F, X>(f, x.f));
      }

      __HD__ Op<Assign<F, C>> operator=(float x) {
        return Op<Assign<F, C>>(Assign<F, C>(f, C(x)));
      }

      template <typename ...Args>
      __HDI__ float operator()(Args&&... args) {
        return f(args...);
      }
    };

    template <int N>
    using ref = Op<X<N>>;

    static ref<1> _1;
    static ref<2> _2;
    static ref<3> _3;
    static ref<4> _4;
    static ref<5> _5;
    static ref<6> _6;
    static ref<7> _7;
    static ref<8> _8;
    static ref<9> _9;

  }
}