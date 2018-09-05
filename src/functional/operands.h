#pragma once

#include <limits>
#include <string>
#include "functional/defs.h"

namespace marian {
namespace functional {

template <class C>
using IsClass = typename std::enable_if<std::is_class<C>::value, C>::type;

template <int N>
struct Select {
  template <typename T, typename... Args>
  __HDI__ static auto apply(T&& /*arg*/, Args&&... args)
      -> decltype(Select<N - 1>::apply(args...)) {
    return Select<N - 1>::apply(args...);
  }
};

template <>
struct Select<0> {
  template <typename T, typename... Args>
  __HDI__ static T apply(T&& arg, Args&&... /*args*/) {
    return arg;
  }
};

/******************************************************************************/

template <int V>
struct C {
  static constexpr auto value = V;

  template <typename... Args>
  __HDI__ float operator()(Args&&... args) {
    return V;
  }

  std::string to_string() { return "C<" + std::to_string(V) + ">"; }
};

/******************************************************************************/

struct Capture {
  float value;

  Capture(float val) : value(val){};

  template <typename... Args>
  __HDI__ float operator()(Args&&... /*args*/) {
    return value;
  }

  std::string to_string() { return "Cap(" + std::to_string(value) + ")"; }
};

/******************************************************************************/

template <int N>
struct Var {
  static constexpr auto index = N;

  template <typename... Args>
  __HDI__ float& operator()(Args&&... args) {
    return Select<N - 1>::apply(args...);
  }

  std::string to_string() { return "Var<" + std::to_string(N) + ">"; }
};
}  // namespace functional
}  // namespace marian
