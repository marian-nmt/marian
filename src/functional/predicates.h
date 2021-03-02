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

  template <typename T, typename... Args>
  HOST_DEVICE_INLINE T operator()(T arg, Args&&... args) {
    return Function::apply(x(arg, args...));
  }

  std::string to_string() const { return Function::n() + "<" + x.to_string() + ">"; }
};

template <class Function, class X, class Y>
struct BinaryFunctor {
  X x;
  Y y;

  template <class Arg1, class Arg2>
  BinaryFunctor(Arg1 arg1, Arg2 arg2) : x(arg1), y(arg2) {}

  template <typename T, typename... Args>
  HOST_DEVICE_INLINE T operator()(T arg, Args&&... args) {
    return Function::apply(x(arg, args...), y(arg, args...));
  }

  std::string to_string() const {
    return Function::n() + "<" + x.to_string() + "," + y.to_string() + ">";
  }
};

/**
 * Macro to set up unary-functions from marian::functional::Ops.
 * @param name name for the struct
 * @param name2 callable typedef
 * @param func function wrapped
 */
#define UNARY(name, name2, func)                                      \
  namespace elem {                                                    \
  struct name {                                                       \
    template <typename ElementType>                                   \
    HOST_DEVICE_INLINE static ElementType apply(const ElementType& x) { return func; } \
    static std::string n() { return #name; }                          \
  };                                                                  \
  }                                                                   \
  template <class X>                                                  \
  using name = UnaryFunctor<elem::name, X>;                           \
  template <typename X>                                               \
  static inline name<IsClass<X>> name2(X x) {                         \
    return name<X>(x);                                                \
  }                                                                   \
  static inline name<Capture> name2(Capture x) { return name<Capture>(x); }

/**
 * Macro to set up binary-functions from marian::functional::Ops.
 * @param name name for the struct
 * @param name2 callable typedef
 * @param func function wrapped
 */
#define BINARY(name, name2, func)                                 \
  namespace elem {                                                \
  struct name {                                                   \
    template <typename ElementType>                               \
    HOST_DEVICE_INLINE static ElementType apply(const ElementType& x,        \
                                                const ElementType& y)        \
      { return func; }                                            \
    static std::string n() { return #name; }                      \
  };                                                              \
  }                                                               \
  template <class X, class Y>                                     \
  using name = BinaryFunctor<elem::name, X, Y>;                   \
  template <class X, class Y>                                     \
  name<IsClass<X>, IsClass<Y>> name2(const X& x, const Y& y) {    \
    return name<X, Y>(x, y);                                      \
  }                                                               \
  template <class Y>                                              \
  name<Capture, IsClass<Y>> name2(const Capture& x, const Y& y) { \
    return name<Capture, Y>(x, y);                                \
  }                                                               \
  template <class X>                                              \
  name<IsClass<X>, Capture> name2(const X& x, const Capture& y) { \
    return name<X, Capture>(x, y);                                \
  }

template <class Function, class X, class Y, class Z>
struct TernaryFunctor {
  X x;
  Y y;
  Z z;

  template <class Arg1, class Arg2, class Arg3>
  TernaryFunctor(Arg1 arg1, Arg2 arg2, Arg3 arg3) : x(arg1), y(arg2), z(arg3) {}

  template <typename T, typename... Args>
  HOST_DEVICE_INLINE T operator()(T arg, Args&&... args) {
    return Function::apply(x(arg, args...), y(arg, args...), z(arg, args...));
  }
};

/**
 * Macro to set up ternary-functions from marian::functional::Ops.
 * @param name name for the struct
 * @param name2 callable typedef
 * @param func function wrapped
 */
#define TERNARY(name, name2, func)                                         \
  namespace elem {                                                         \
  struct name {                                                            \
    template <typename ElementType>                                        \
    HOST_DEVICE_INLINE static ElementType apply(ElementType x,             \
                                                ElementType y,             \
                                                ElementType z)             \
    { return func; }                                                       \
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

template <class X, class Y>
struct Assign {
  X x;
  Y y;

  template <class Arg1, class Arg2>
  Assign(Arg1 arg1, Arg2 arg2) : x(arg1), y(arg2) {}

  template <typename T, typename... Args>
  HOST_DEVICE_INLINE T operator()(T&& arg, Args&&... args) {
    return x(arg, args...) = y(arg, args...);
  }

  std::string to_string() const {
    return "Assign<" + x.to_string() + "," + y.to_string() + ">";
  }
};

template <int N>
struct Assignee {
  Var<N> var;

  Assignee() {}
  Assignee(Var<N> v) : var(v) {}

  template <typename T, typename... Args>
  HOST_DEVICE_INLINE T& operator()(T&& arg, Args&&... args) {
    return var(arg, args...);
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

  std::string to_string() const { return var.to_string(); }
};

/******************************************************************************/
}  // namespace functional
}  // namespace marian
