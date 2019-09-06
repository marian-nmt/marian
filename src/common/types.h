#pragma once

#include "half_float/umHalf.h"

#include <iostream>
#include <string>
#include <functional>

#ifndef __CUDA_ARCH__
#include <immintrin.h>
#endif

#define DISPATCH_BY_TYPE0(type, func) \
do { \
  switch(type) { \
    case Type::int8:    return func<int8_t  >(); \
    case Type::int16:   return func<int16_t >(); \
    case Type::int32:   return func<int32_t >(); \
    case Type::int64:   return func<int64_t >(); \
    case Type::uint8:   return func<uint8_t >(); \
    case Type::uint16:  return func<uint16_t>(); \
    case Type::uint32:  return func<uint32_t>(); \
    case Type::uint64:  return func<uint64_t>(); \
    case Type::float16: return func<float16 >(); \
    case Type::float32: return func<float   >(); \
    case Type::float64: return func<double  >(); \
    default: ABORT("Unknown type {}", type); \
  } \
} while(0)

#define DISPATCH_BY_TYPE1(type, func, arg1) \
do { \
  switch(type) { \
    case Type::int8:    return func<int8_t  >(arg1); \
    case Type::int16:   return func<int16_t >(arg1); \
    case Type::int32:   return func<int32_t >(arg1); \
    case Type::int64:   return func<int64_t >(arg1); \
    case Type::uint8:   return func<uint8_t >(arg1); \
    case Type::uint16:  return func<uint16_t>(arg1); \
    case Type::uint32:  return func<uint32_t>(arg1); \
    case Type::uint64:  return func<uint64_t>(arg1); \
    case Type::float16: return func<float16 >(arg1); \
    case Type::float32: return func<float   >(arg1); \
    case Type::float64: return func<double  >(arg1); \
    default: ABORT("Unknown type {}", type); \
  } \
} while(0)

#define DISPATCH_BY_TYPE2(type, func, arg1, arg2) \
do { \
  switch(type) { \
    case Type::int8    : return func<int8_t  >(arg1, arg2); \
    case Type::int16   : return func<int16_t >(arg1, arg2); \
    case Type::int32   : return func<int32_t >(arg1, arg2); \
    case Type::int64   : return func<int64_t >(arg1, arg2); \
    case Type::uint8   : return func<uint8_t >(arg1, arg2); \
    case Type::uint16  : return func<uint16_t>(arg1, arg2); \
    case Type::uint32  : return func<uint32_t>(arg1, arg2); \
    case Type::uint64  : return func<uint64_t>(arg1, arg2); \
    case Type::float16 : return func<float16 >(arg1, arg2); \
    case Type::float32 : return func<float   >(arg1, arg2); \
    case Type::float64 : return func<double  >(arg1, arg2); \
    default: ABORT("Unknown type {}", type); \
  } \
} while(0)

namespace marian {

#ifndef __CUDA_ARCH__
struct float32x4 {
private:
  __m128 f_;

public:
  float32x4() {}
  float32x4(const __m128& f) : f_(f) {}
  float32x4(const float& f) : f_(_mm_set1_ps(f)) {}

  operator const __m128&() const { return f_; }
  operator __m128&() { return f_; }

  float operator[] (size_t i) const {
    return *(((float*)&f_) + i);
  }

  friend std::ostream& operator<<(std::ostream& out, float32x4 f4) {
    float* a = (float*)&f4;
    out << "[" << a[0];
    for(int i = 1; i < 4; i++)
      out << " " << a[i];
    out << "]";
    return out;
  }
};

struct float32x8 {
private:
  __m256 f_;

public:
  float32x8() {}
  float32x8(const __m256& f) : f_(f) {}
  float32x8(const float& f) : f_(_mm256_set1_ps(f)) {}

  operator const __m256&() const { return f_; }
  operator __m256&() { return f_; }

  float operator[] (size_t i) const {
    return *(((float*)&f_) + i);
  }

  friend std::ostream& operator<<(std::ostream& out, float32x8 f8) {
    float* a = (float*)&f8;
    out << "[" << a[0];
    for(int i = 1; i < 8; i++)
      out << " " << a[i];
    out << "]";
    return out;
  }
};
#endif

enum class TypeClass : size_t {
  signed_type = 0x100,
  unsigned_type = 0x200,
  float_type = 0x400,
  size_mask = 0x0FF
};

constexpr inline size_t operator+(TypeClass typeClass, size_t val) {
  return (size_t)typeClass + val;
}

enum class Type : size_t {
  int8  = TypeClass::signed_type + 1u,
  int16 = TypeClass::signed_type + 2u,
  int32 = TypeClass::signed_type + 4u,
  int64 = TypeClass::signed_type + 8u,

  uint8  = TypeClass::unsigned_type + 1u,
  uint16 = TypeClass::unsigned_type + 2u,
  uint32 = TypeClass::unsigned_type + 4u,
  uint64 = TypeClass::unsigned_type + 8u,

  float16 = TypeClass::float_type + 2u,
  float32 = TypeClass::float_type + 4u,
  float64 = TypeClass::float_type + 8u
};

static inline size_t operator&(TypeClass typeClass, Type type) {
  return (size_t)typeClass & (size_t)type;
}

static inline size_t sizeOf(Type type) {
  return TypeClass::size_mask & type;
}

static inline bool isSignedInt(Type type) {
  return (TypeClass::signed_type & type) != 0;
}

static inline bool isUnsignedInt(Type type) {
  return (TypeClass::unsigned_type & type) != 0;
}

static inline bool isInt(Type type) {
  return isSignedInt(type) || isUnsignedInt(type);
}

static inline bool isFloat(Type type) {
  return (TypeClass::float_type & type) != 0;
}

template <typename T>
inline bool matchType(Type type);

// clang-format off
template <> inline bool matchType<int8_t>(Type type)   { return type == Type::int8; }
template <> inline bool matchType<int16_t>(Type type)  { return type == Type::int16; }
template <> inline bool matchType<int32_t>(Type type)  { return type == Type::int32; }
template <> inline bool matchType<int64_t>(Type type)  { return type == Type::int64; }

template <> inline bool matchType<uint8_t>(Type type)  { return type == Type::uint8; }
template <> inline bool matchType<uint16_t>(Type type) { return type == Type::uint16; }
template <> inline bool matchType<uint32_t>(Type type) { return type == Type::uint32; }
template <> inline bool matchType<uint64_t>(Type type) { return type == Type::uint64; }

template <> inline bool matchType<float16>(Type type)  { return type == Type::float16; }
template <> inline bool matchType<float>(Type type)    { return type == Type::float32; }
template <> inline bool matchType<double>(Type type)   { return type == Type::float64; }
// clang-format on

static inline std::ostream& operator<<(std::ostream& out, Type type) {
  switch(type) {
    case Type::int8    : out << "int8"; break;
    case Type::int16   : out << "int16"; break;
    case Type::int32   : out << "int32"; break;
    case Type::int64   : out << "int64"; break;

    case Type::uint8   : out << "uint8"; break;
    case Type::uint16  : out << "uint16"; break;
    case Type::uint32  : out << "uint32"; break;
    case Type::uint64  : out << "uint64"; break;

    case Type::float16 : out << "float16"; break;
    case Type::float32 : out << "float32"; break;
    case Type::float64 : out << "float64"; break;
  }
  return out;
}

template <typename T>
inline std::string request();

// clang-format off
template <> inline std::string request<int8_t>()  { return "int8"; }
template <> inline std::string request<int16_t>() { return "int16"; }
template <> inline std::string request<int32_t>() { return "int32"; }
template <> inline std::string request<int64_t>() { return "int64"; }

template <> inline std::string request<uint8_t>()  { return "uint8"; }
template <> inline std::string request<uint16_t>() { return "uint16"; }
template <> inline std::string request<uint32_t>() { return "uint32"; }
template <> inline std::string request<uint64_t>() { return "uint64"; }

template <> inline std::string request<float16>()  { return "float16"; }
template <> inline std::string request<float>()    { return "float32"; }
template <> inline std::string request<double>()   { return "float64"; }
// clang-format on

static Type inline typeFromString(const std::string& str) {
  if(str == "int8")
    return Type::int8;
  if(str == "int16")
    return Type::int16;
  if(str == "int32")
    return Type::int32;
  if(str == "int64")
    return Type::int64;

  if(str == "uint8")
    return Type::uint8;
  if(str == "uint16")
    return Type::uint16;
  if(str == "uint32")
    return Type::uint32;
  if(str == "uint64")
    return Type::uint64;

  if(str == "float16")
    return Type::float16;
  if(str == "float32")
    return Type::float32;
  if(str == "float64")
    return Type::float64;

  ABORT("Unknown type {}", str);
}

template <typename T>
inline Type typeId();

template <> inline Type typeId<int8_t>()   { return Type::int8; }
template <> inline Type typeId<int16_t>()  { return Type::int16; }
template <> inline Type typeId<int32_t>()  { return Type::int32; }
template <> inline Type typeId<int64_t>()  { return Type::int64; }

template <> inline Type typeId<uint8_t>()  { return Type::uint8; }
template <> inline Type typeId<uint16_t>() { return Type::uint16; }
template <> inline Type typeId<uint32_t>() { return Type::uint32; }
template <> inline Type typeId<uint64_t>() { return Type::uint64; }

template <> inline Type typeId<float16>()  { return Type::float16; }
template <> inline Type typeId<float>()    { return Type::float32; }
template <> inline Type typeId<double>()   { return Type::float64; }

// Abort if given C++ does not correspond to runtime type
template <typename T>
void matchOrAbort(Type type) {
  // @TODO: hacky hack for WGNMT, turn this back on
  // ABORT_IF(!matchType<T>(type),
  //          "Requested type ({}) and underlying type ({}) do not match",
  //          request<T>(),
  //          type);
}

template <typename T>
class NumericLimits {
private:
  template <typename L>
  void setLimits() {
    max = std::numeric_limits<L>::max();
    min = std::numeric_limits<L>::min();
    lowest = std::numeric_limits<L>::lowest();
  }

  void setLimits(Type type) {
    DISPATCH_BY_TYPE0(type, setLimits);
  }

public:
  T max;
  T min;
  T lowest;

  NumericLimits(Type type) {
    setLimits(type);
  }
};

}  // namespace marian
