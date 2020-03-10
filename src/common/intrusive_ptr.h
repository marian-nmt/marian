#pragma once

#include <cassert>
#include <iostream>
#include "common/logging.h"

// Smart pointer class for small objects with reference counting but no thread-safety.
// Inspired by boost::intrusive_ptr<T>.

// Compared to std::shared_ptr this is small and cheap to construct and destroy.
// Does not hold the counter, the pointed to class `T` needs to add
// ENABLE_INTRUSIVE_PTR(T) into the body of the class (private section). This adds
// the reference counters and count manipulation functions to the class.

#define ENABLE_INTRUSIVE_PTR(type)           \
  size_t references_{0};                     \
                                             \
  inline friend void intrusivePtrAddRef(type* x) {  \
    if(x != 0)                               \
      ++x->references_;                      \
  }                                          \
                                             \
  inline friend void intrusivePtrRelease(type* x) { \
    if(x != 0 && --x->references_ == 0) {    \
      delete x;                              \
      x = 0;                                 \
    }                                        \
  }                                          \
                                             \
  inline friend size_t references(type* x) {        \
    return x->references_;                   \
  }                                          \


template<class T>
class IntrusivePtr {
private:
  typedef IntrusivePtr this_type;

public:
  typedef T element_type;

  IntrusivePtr() : ptr_(0) {};

  IntrusivePtr(T* p)
  : ptr_(p) {
      if(ptr_ != 0)
        intrusivePtrAddRef(ptr_);
  }

  template<class Y>
  IntrusivePtr(const IntrusivePtr<Y>& rhs)
  : ptr_(rhs.get()) {
    if(ptr_ != 0)
      intrusivePtrAddRef(ptr_);
  }

  IntrusivePtr(const IntrusivePtr& rhs)
  : ptr_(rhs.ptr_) {
    if(ptr_ != 0)
      intrusivePtrAddRef(ptr_);
  }

  ~IntrusivePtr() {
    if(ptr_ != 0)
      intrusivePtrRelease(ptr_);
  }

  IntrusivePtr(IntrusivePtr&& rhs)
  : ptr_(rhs.ptr_) {
    rhs.ptr_ = 0;
  }

  inline size_t useCount() {
    return references(ptr_);
  }

  inline IntrusivePtr& operator=(IntrusivePtr&& rhs) {
    this_type(static_cast<IntrusivePtr&&>(rhs)).swap(*this);
    return *this;
  }

  inline IntrusivePtr& operator=(const IntrusivePtr& rhs) {
    this_type(rhs).swap(*this);
    return *this;
  }

  template<class Y>
  inline IntrusivePtr& operator=(const IntrusivePtr<Y>& rhs) {
    this_type(rhs).swap(*this);
    return *this;
  }

  inline void reset() {
    this_type().swap(*this);
  }

  inline void reset(T* rhs) {
    this_type(rhs).swap(*this);
  }

  inline T* get() const {
    return ptr_;
  }

  inline T* detach() {
    T* ret = ptr_;
    ptr_ = 0;
    return ret;
  }

  inline T& operator*() const {
    //ABORT_IF(ptr_ == 0, "Null pointer in IntrusivePtr");
    return *ptr_;
  }

  inline T* operator->() const {
    //ABORT_IF(ptr_ == 0, "Null pointer in IntrusivePtr");
    return ptr_;
  }

  inline explicit operator bool() const {
    return ptr_ != 0;
  }

  inline bool operator!() const {
    return ptr_ == 0;
  }

  inline void swap(IntrusivePtr& rhs) {
    T* tmp = ptr_;
    ptr_ = rhs.ptr_;
    rhs.ptr_ = tmp;
  }

private:
  T* ptr_;
};

template<class T, class U>
inline bool operator==(const IntrusivePtr<T>& a, const IntrusivePtr<U>& b) {
  return a.get() == b.get();
}

template<class T, class U>
inline bool operator!=(const IntrusivePtr<T>& a, const IntrusivePtr<U>& b) {
  return a.get() != b.get();
}

template<class T>
inline bool operator==(const IntrusivePtr<T>& a, std::nullptr_t) {
  return a.get() == 0;
}

template<class T>
inline bool operator!=(const IntrusivePtr<T>& a, std::nullptr_t) {
  return a.get() != 0;
}

template<class T>
inline bool operator==(const IntrusivePtr<T>& a, T* b) {
  return a.get() == b;
}

template<class T>
inline bool operator!=(const IntrusivePtr<T>& a, T* b) {
  return a.get() != b;
}

template<class T>
inline bool operator==(T* a, const IntrusivePtr<T>& b) {
  return a == b.get();
}

template<class T>
inline bool operator!=(T* a, const IntrusivePtr<T>& b) {
  return a != b.get();
}

template<class T, class U>
inline bool operator<(const IntrusivePtr<T>& a, const IntrusivePtr<U>& b) {
  return std::less<T*>()(a.get(), b.get());
}

template<class T>
inline void swap(IntrusivePtr<T> & a, IntrusivePtr<T> & b) {
  a.swap(b);
}

template<class E, class T, class Y>
std::basic_ostream<E, T>& operator<<(std::basic_ostream<E, T>& os, const IntrusivePtr<Y>& p) {
  os << p.get();
  return os;
}

// compatibility functions to make std::*_pointer_cast<T> work, also for automatic hashing
namespace std {
  template<class T>
  T* get_pointer(const IntrusivePtr<T>& p) {
    return p.get();
  }

  template<class T, class U>
  IntrusivePtr<T> static_pointer_cast(const IntrusivePtr<U>& p) {
    return static_cast<T*>(p.get());
  }

  template<class T, class U>
  IntrusivePtr<T> const_pointer_cast(const IntrusivePtr<U>& p) {
    return const_cast<T*>(p.get());
  }

  template<class T, class U>
  IntrusivePtr<T> dynamic_pointer_cast(const IntrusivePtr<U>& p) {
    return dynamic_cast<T*>(p.get());
  }

  // IntrusivePtr<T> can be used as hash map key
  template <class T> struct hash<IntrusivePtr<T>> {
    size_t operator()(const IntrusivePtr<T>& x) const {
      std::hash<size_t> hasher;
      return hasher((size_t)x.get());
    }
  };
}
