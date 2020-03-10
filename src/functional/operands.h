#pragma once

#include "functional/defs.h"
#include <limits>
#include <string>

namespace marian {
namespace functional {

template <class C>
using IsClass = typename std::enable_if<std::is_class<C>::value, C>::type;

template <int N>
struct Select {
  template <typename T, typename... Args>
  HOST_DEVICE_INLINE static auto apply(T&& /*arg*/, Args&&... args)
      -> decltype(Select<N - 1>::apply(args...)) {
    return Select<N - 1>::apply(args...);
  }
};

template <>
struct Select<0> {
  template <typename T, typename... Args>
  HOST_DEVICE_INLINE static T apply(T&& arg, Args&&... /*args*/) {
    return arg;
  }
};

/******************************************************************************/

template <int V>
struct C {
  static constexpr auto value = V;

  template <typename T, typename... Args>
  HOST_DEVICE_INLINE T operator()(T&& /*arg*/, Args&&... /*args*/) {
    return (T)V;
  }

  std::string to_string() const { return "C<" + std::to_string(V) + ">"; }
};

/******************************************************************************/

struct Capture {
  float value;

  Capture(float val) : value(val){};

  template <typename T, typename... Args>
  HOST_DEVICE_INLINE T operator()(const T& /*arg*/, const Args&... /*args*/) {
    return T(value);
  }

  std::string to_string() const { return "Cap(" + std::to_string(value) + ")"; }
};

/******************************************************************************/

template <int N>
struct Var {
  static constexpr auto index = N;

  template <typename T, typename... Args>
  HOST_DEVICE_INLINE T& operator()(T&& arg, Args&&... args) {
    return Select<N - 1>::apply(arg, args...);
  }

  std::string to_string() const { return "Var<" + std::to_string(N) + ">"; }
};
}  // namespace functional
}  // namespace marian
