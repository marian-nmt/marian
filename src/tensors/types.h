#pragma once

#include <iostream>
#include <string>

namespace marian {

enum class TypeClass : size_t {
    signed_type   = 0x100,
    unsigned_type = 0x200,
    float_type    = 0x400,
    size_mask     = 0x0FF
};

constexpr inline size_t operator+(TypeClass typeClass, size_t val) {
    return (size_t)typeClass + val;
}

enum class Type : size_t {
  int8    = TypeClass::signed_type   + 1u,
  int16   = TypeClass::signed_type   + 2u,
  int32   = TypeClass::signed_type   + 4u,
  int64   = TypeClass::signed_type   + 8u,

  uint8   = TypeClass::unsigned_type + 1u,
  uint16  = TypeClass::unsigned_type + 2u,
  uint32  = TypeClass::unsigned_type + 4u,
  uint64  = TypeClass::unsigned_type + 8u,

  float32 = TypeClass::float_type    + 4u,
  float64 = TypeClass::float_type    + 8u
};

static inline size_t operator&(TypeClass typeClass, Type type) {
    return (size_t)typeClass & (size_t)type;
}

static inline size_t sizeOf(Type type) {
  return TypeClass::size_mask & type;
}

static inline bool isSignedInt(Type type) {
    return TypeClass::signed_type & type;
}

static inline bool isUnsignedInt(Type type) {
    return TypeClass::unsigned_type & type;
}

static inline bool isInt(Type type) {
    return isSignedInt(type) || isUnsignedInt(type);
}

static inline bool isFloat(Type type) {
    return TypeClass::float_type & type;
}

template <typename T>
inline bool matchType(Type type);

template <> inline bool matchType<int8_t>(Type type)  { return type == Type::int8; }
template <> inline bool matchType<int16_t>(Type type) { return type == Type::int16; }
template <> inline bool matchType<int32_t>(Type type) { return type == Type::int32; }
template <> inline bool matchType<int64_t>(Type type) { return type == Type::int64; }

template <> inline bool matchType<uint8_t>(Type type)  { return type == Type::uint8; }
template <> inline bool matchType<uint16_t>(Type type) { return type == Type::uint16; }
template <> inline bool matchType<uint32_t>(Type type) { return type == Type::uint32; }
template <> inline bool matchType<uint64_t>(Type type) { return type == Type::uint64; }

template <> inline bool matchType<float>(Type type)  { return type == Type::float32; }
template <> inline bool matchType<double>(Type type) { return type == Type::float64; }

static inline std::ostream& operator<<(std::ostream& out, Type type) {
    switch (type) {
      case Type::int8:  out << "int8"; break;
      case Type::int16: out << "int16"; break;
      case Type::int32: out << "int32"; break;
      case Type::int64: out << "int64"; break;

      case Type::uint8:  out << "uint8"; break;
      case Type::uint16: out << "uint16"; break;
      case Type::uint32: out << "uint32"; break;
      case Type::uint64: out << "uint64"; break;

      case Type::float32: out << "float32"; break;
      case Type::float64: out << "float64"; break;
    }
    return out;
}

template <typename T>
inline std::string request();

template <> inline std::string request<int8_t>()  { return "int8"; }
template <> inline std::string request<int16_t>() { return "int16"; }
template <> inline std::string request<int32_t>() { return "int32"; }
template <> inline std::string request<int64_t>() { return "int64"; }

template <> inline std::string request<uint8_t>()  { return "uint8"; }
template <> inline std::string request<uint16_t>() { return "uint16"; }
template <> inline std::string request<uint32_t>() { return "uint32"; }
template <> inline std::string request<uint64_t>() { return "uint64"; }

template <> inline std::string request<float>()  { return "float32"; }
template <> inline std::string request<double>() { return "float64"; }

}
