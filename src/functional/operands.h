#pragma once

#include "gpu/defs.h"

namespace marian {
  namespace functional {

    template <class C>
    using IsClass = typename std::enable_if<std::is_class<C>::value, C>::type;

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

/******************************************************************************/

    template <int V>
    struct C {
      template <typename ...Args>
      __HDI__ float operator()(Args&&... args) { return V; }
    };

/******************************************************************************/

    struct Capture {
      float value;

      Capture(float val) : value(val) {};

      template <typename ...Args>
      __HDI__ float operator()(Args&&... args) { return value; }
    };

/******************************************************************************/

    template <int N>
    struct Var {

      template <typename ...Args>
      __HDI__ float& operator()(Args&&... args) {
        return Select<N-1>::apply(args...);
      }
    };

  }
}