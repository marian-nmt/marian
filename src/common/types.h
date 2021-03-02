#pragma once
#include "common/logging.h" // for ABORT and ABORT_IF
#include "common/shape.h"

#if __GNUC__ >= 7
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wint-in-bool-context" // gcc-7 introduces this warning, triggered in 3rd-party code
#endif
#include "half_float/umHalf.h"
#if __GNUC__ >= 7
#pragma GCC diagnostic pop
#endif

#include <iostream>
#include <string>
#include <functional>
#include <type_traits>

#ifndef __CUDACC__ // NVCC is very unreliable when it comes to CPU intrinsics, we hide them completely from NVCC-compiled code
#include <immintrin.h>
#endif

#ifdef __CUDACC__ // nvcc is compiling this code
#include <cuda.h> // required to see CUDA_VERSION
#if (CUDA_VERSION > 9000 && (__CUDA_ARCH__ >= 600 || !defined(__CUDA_ARCH__)))
#define COMPILE_FP16 1 // we are in GPU code and we know what to do with FP16 code
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable:4505) // "unreferenced local function has been removed" in cuda_fp16.hpp
#endif
#include <cuda_fp16.h>
#include "functional/defs.h"
#else
#define COMPILE_FP16 0 // we are in GPU code, but compute capability is too low to use FP16
#endif
#elif CUDA_FOUND // other compiler, likely host code. Should be fine with seeing the correct includes with host code
#include <cuda.h> // required to see CUDA_VERSION
#if (CUDA_VERSION > 9000)
#define COMPILE_FP16 1
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable:4505) // "unreferenced local function has been removed" in cuda_fp16.hpp
#endif
#include <cuda_fp16.h>
#include "functional/defs.h"
#else
#define COMPILE_FP16 0
#endif
#else
#define COMPILE_FP16 0
#endif

#ifdef _MSC_VER
// @BUGBUG: Visual Studio somehow fails on template expansions for float16.
//          To be able to build on Windows, we temporarily disable this, until the greater merge has happened.
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
    case Type::float16: ABORT("Broken type {}", type);/*return func<float16 >();*/ \
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
    case Type::float16: ABORT("Broken type {}", type);/*return func<float16 >(arg1);*/ \
    case Type::float32: return func<float   >(arg1); \
    case Type::float64: return func<double  >(arg1); \
    default: ABORT("Unknown type {}", type); \
  } \
} while(0)
#else
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
#endif

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
/// namespace marian
namespace marian {

// small struct to enable templating based on types use for packing
struct packed16 { uint16_t x; };

// small struct to enable templating based on types use for packing. This is a memory holder.
// There's no difference between packed8avx2 and packed8avx512. But, they are separately defined to be distinguished.
struct packed8avx2   { uint8_t x; };
struct packed8avx512 { uint8_t x; };

// similar to the packed16, but to use with 16bit intgemm model packing.
struct intgemm16       { int16_t x; };
struct intgemm16sse2   { int16_t x; };
struct intgemm16avx2   { int16_t x; };
struct intgemm16avx512 { int16_t x; };

// similar to packed8* but for intgemm 8bit model packing.
struct intgemm8            { int8_t x;  };
struct intgemm8ssse3       { int8_t x;  };
struct intgemm8avx2        { int8_t x;  };
struct intgemm8avx512      { int8_t x;  };
struct intgemm8avx512vnni  { int8_t x;  };


#ifndef __CUDACC__ // vectorized types not available from .cu files

// @TODO: check what intrinsics are actually available.
struct float32x4 {
private:
  __m128 f_;

public:
  float32x4() {}
  float32x4(const __m128& f) : f_(f) {}
  float32x4(const float& f) : f_(_mm_set1_ps(f)) {} // __m128 _mm_set1_ps(float) copies value into all slots

  operator const __m128&() const { return f_; }
  operator __m128&() { return f_; }

  float operator[] (size_t i) const {
    return *(((float*)&f_) + i); // potentially undefined, but efficient. In practice __m128 is an array of floats
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

// @TODO: consider how code can be shared via templating
#ifdef __AVX__
struct float32x8 {
private:
  __m256 f_;

public:
  float32x8() {}
  float32x8(const __m256& f) : f_(f) {}
  float32x8(const float& f) : f_(_mm256_set1_ps(f)) {} // __m256 _mm_set1_ps(float) copies value into all slots

  operator const __m256&() const { return f_; }
  operator __m256&() { return f_; }

  float operator[] (size_t i) const {
    return *(((float*)&f_) + i); // potentially undefined, but efficient. In practice __m128 is an array of floats
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
#else
//Dummy version to get things to compile on older CPUs
struct float32x8 {
};
#endif
#endif

#if COMPILE_FP16

// @TODO: check what intrinsics are actually available.
struct halfx2 {
private:
  __half2 h2_;

public:
  DEVICE halfx2() {}
  DEVICE halfx2(const __half2& h2) : h2_(h2) {}
  DEVICE halfx2(const __half& h) : h2_(h, h) {}
  DEVICE halfx2(const __half& h1, const __half& h2) : h2_(h1, h2) {}

  DEVICE_INLINE operator const __half2&() const { return h2_; }
  DEVICE_INLINE operator __half2&() { return h2_; }

  DEVICE_INLINE __half operator[] (size_t i) const {
    return *(((__half*)&h2_) + i); // potentially undefined, but efficient. In practice __m128 is an array of floats
  }

  friend std::ostream& operator<<(std::ostream& out, halfx2 h2) {
    __half* a = (__half*)&h2;
    out << "[" << (float)a[0];
    for(int i = 1; i < 2; i++)
      out << " " << (float)a[i];
    out << "]";
    return out;
  }
};
#endif

// Internal to types.h, don't use. Use test functions below.
enum class TypeClass : size_t { // size_type has 8 bytes, so we can have 16 fields here, currently using 5. Extend to the left for back-compat.
  // built-in type classes
  signed_type   = 0x00100,
  unsigned_type = 0x00200,
  float_type    = 0x00400,

  avx2_type     = 0x01000, // processor-specific layout for avx2, currently used for FBGEMM only (keep 0x1000 for back-compat)
  avx512_type   = 0x02000, // processor-specific layout for avx512, currently used for FBGEMM only (keep 0x2000 for back-compat)
  sse2_type     = 0x04000, // processor-specific layout for sse2, currently used for Intgemm only
  ssse3_type    = 0x08000, // processor-specific layout for ssse3, currently used for Intgemm only

  packed_type   = 0x00800, // special packed (CPU cache friendly) type class, used in FBGEMM. Annoyingly we need to keep 0x800 for back-compat, would be nicer to align with intgemm
  intgemm_type  = 0x10000, // intgemm quantized architecture agnostic models

  size_mask     = 0x000FF, // maximum allowed size is 256 bytes right now; if more are required, extend the size field
  class_mask    = 0xFFF00, // three fields for different type classes, if more classes are added we need to increase the number of fields here
};

constexpr inline size_t operator+(TypeClass typeClass, size_t val) {
  return (size_t)typeClass + val;
}

constexpr inline size_t operator+(size_t val, TypeClass typeClass) {
  return val + (size_t)typeClass;
}

// @TODO: rename to ElementType when things become stable, so it's easier to review
/// enum class Type: stores all supported data type in Marian
enum class Type : size_t {
  int8     = TypeClass::signed_type + 1u,      ///< int8 type
  int16    = TypeClass::signed_type + 2u,      ///< int16 type
  int32    = TypeClass::signed_type + 4u,      ///< int32 type
  int64    = TypeClass::signed_type + 8u,      ///< int64 type

  uint8    = TypeClass::unsigned_type + 1u,    ///< uint8 type
  uint16   = TypeClass::unsigned_type + 2u,    ///< uint16 type
  uint32   = TypeClass::unsigned_type + 4u,    ///< uint32 type
  uint64   = TypeClass::unsigned_type + 8u,    ///< uint64 type

  float16  = TypeClass::float_type + 2u,       ///< float16 type
  float32  = TypeClass::float_type + 4u,       ///< float32 type
  float64  = TypeClass::float_type + 8u,       ///< float64 type

  packed16            = TypeClass::packed_type + 2u,                                   ///< special type for FBGEMM, not meant to be used anywhere else, not meant to be accessed invidually. Internal actual type (uint16) is meaningless.
  packed8avx2         = TypeClass::packed_type + 1u + TypeClass::avx2_type,            ///< special type for FBGEMM with AVX2, not meant to be used anywhere else, not meant to be accessed invidually. Internal actual type (uint8) is meaningless.
  packed8avx512       = TypeClass::packed_type + 1u + TypeClass::avx512_type,          ///< special type for FBGEMM with AVX512, not meant to be used anywhere else, not meant to be accessed invidually. Internal actual type (uint8) is meaningless.

  intgemm8            = TypeClass::intgemm_type + 1u,                                  ///< Int8 quantized (not packed) matrices for intgemm
  intgemm16           = TypeClass::intgemm_type + 2u,                                  ///< Int16 quantized (not packed) matrices for intgemm
  
  intgemm8ssse3       = TypeClass::intgemm_type + 1u + TypeClass::ssse3_type,          ///< Int8 quantized and packed (ssse3) matrices for intgemm
  intgemm8avx2        = TypeClass::intgemm_type + 1u + TypeClass::avx2_type,           ///< Int8 quantized and packed (avx2) matrices for intgemm
  intgemm8avx512      = TypeClass::intgemm_type + 1u + TypeClass::avx512_type,         ///< Int8 quantized and packed (avx512) matrices for intgemm
  intgemm8avx512vnni  = TypeClass::intgemm_type + 1u + TypeClass::avx512_type + 4096u, ///< Int8 quantized and packed (avx512) matrices for intgemm. VNNI algorithm

  intgemm16sse2       = TypeClass::intgemm_type + 2u + TypeClass::sse2_type,           ///< Int16 quantized and packed (sse2) matrices for intgemm
  intgemm16avx2       = TypeClass::intgemm_type + 2u + TypeClass::avx2_type,           ///< Int16 quantized and packed (avx2) matrices for intgemm
  intgemm16avx512     = TypeClass::intgemm_type + 2u + TypeClass::avx512_type,         ///< Int16 quantized and packed (avx512) matrices for intgemm
};

static inline size_t operator&(TypeClass typeClass, Type type) {
  return (size_t)typeClass & (size_t)type;
}

static inline bool isSameTypeClass(Type type1, Type type2) {
  return (TypeClass::class_mask & type1) == (TypeClass::class_mask & type2);
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

static inline bool isPacked(Type type) {
  return (TypeClass::packed_type & type) != 0;
}

static inline bool isSse2(Type type) {
  return (TypeClass::sse2_type & type) != 0;
}

static inline bool isSsse3(Type type) {
  return (TypeClass::ssse3_type & type) != 0;
}

static inline bool isAvx2(Type type) {
  return (TypeClass::avx2_type & type) != 0;
}

static inline bool isAvx512(Type type) {
  return (TypeClass::avx512_type & type) != 0;
}

static inline bool isIntgemm(Type type) {
  return (TypeClass::intgemm_type & type) != 0;
}

size_t requiredBytes(const Shape& shape, Type type); // towards Frank's vision of joint Shape/Type

template <typename T>
inline bool matchType(Type type);

// clang-format off
template <> inline bool matchType<int8_t>(Type type)   { return type == Type::int8;     }
template <> inline bool matchType<int16_t>(Type type)  { return type == Type::int16;    }
template <> inline bool matchType<int32_t>(Type type)  { return type == Type::int32;    }
template <> inline bool matchType<int64_t>(Type type)  { return type == Type::int64;    }

// In case of packed type, it uses uint8 as underlying memory type
template <> inline bool matchType<uint8_t>(Type type)  { return type == Type::uint8;    }
template <> inline bool matchType<uint16_t>(Type type) { return type == Type::uint16;   }
template <> inline bool matchType<uint32_t>(Type type) { return type == Type::uint32;   }
template <> inline bool matchType<uint64_t>(Type type) { return type == Type::uint64;   }

template <> inline bool matchType<float16>(Type type)              { return type == Type::float16;             }
template <> inline bool matchType<float>(Type type)                { return type == Type::float32;             }
template <> inline bool matchType<double>(Type type)               { return type == Type::float64;             }

template <> inline bool matchType<packed16>(Type type)             { return type == Type::packed16;            }
template <> inline bool matchType<packed8avx2>(Type type)          { return type == Type::packed8avx2;         }
template <> inline bool matchType<packed8avx512>(Type type)        { return type == Type::packed8avx512;       }

template <> inline bool matchType<intgemm8>(Type type)             { return type == Type::intgemm8;            }
template <> inline bool matchType<intgemm8ssse3>(Type type)        { return type == Type::intgemm8ssse3;       }
template <> inline bool matchType<intgemm8avx2>(Type type)         { return type == Type::intgemm8avx2;        }
template <> inline bool matchType<intgemm8avx512>(Type type)       { return type == Type::intgemm8avx512;      }
template <> inline bool matchType<intgemm8avx512vnni>(Type type)   { return type == Type::intgemm8avx512vnni;  }

template <> inline bool matchType<intgemm16>(Type type)            { return type == Type::intgemm16;           }
template <> inline bool matchType<intgemm16sse2>(Type type)        { return type == Type::intgemm16sse2;       }
template <> inline bool matchType<intgemm16avx2>(Type type)        { return type == Type::intgemm16avx2;       }
template <> inline bool matchType<intgemm16avx512>(Type type)      { return type == Type::intgemm16avx512;     }
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

    case Type::packed16      : out << "packed16"; break;
    case Type::packed8avx2   : out << "packed8avx2"; break;
    case Type::packed8avx512 : out << "packed8avx512"; break;

    case Type::intgemm8            : out << "intgemm8"; break;
    case Type::intgemm8ssse3       : out << "intgemm8ssse3"; break;
    case Type::intgemm8avx2        : out << "intgemm8avx2"; break;
    case Type::intgemm8avx512      : out << "intgemm8avx512"; break;
    case Type::intgemm8avx512vnni  : out << "intgemm8avx512vnni"; break;
    case Type::intgemm16           : out << "intgemm16"; break;
    case Type::intgemm16sse2       : out << "intgemm16sse2"; break;
    case Type::intgemm16avx2       : out << "intgemm16avx2"; break;
    case Type::intgemm16avx512     : out << "intgemm16avx512"; break;
  }
  return out;
}

template <typename T>
inline std::string request();

// clang-format off
template <> inline std::string request<int8_t>()  { return "int8";  }
template <> inline std::string request<int16_t>() { return "int16"; }
template <> inline std::string request<int32_t>() { return "int32"; }
template <> inline std::string request<int64_t>() { return "int64"; }

template <> inline std::string request<uint8_t>()  { return "uint8";  }
template <> inline std::string request<uint16_t>() { return "uint16"; }
template <> inline std::string request<uint32_t>() { return "uint32"; }
template <> inline std::string request<uint64_t>() { return "uint64"; }

template <> inline std::string request<float16>()  { return "float16"; }
template <> inline std::string request<float>()    { return "float32"; }
template <> inline std::string request<double>()   { return "float64"; }

template <> inline std::string request<packed16>()      { return "packed16";      }
template <> inline std::string request<packed8avx2>()   { return "packed8avx2";   }
template <> inline std::string request<packed8avx512>() { return "packed8avx512"; }

template <> inline std::string request<intgemm8>()            { return "intgemm8";        }
template <> inline std::string request<intgemm8ssse3>()       { return "intgemm8ssse3";    }
template <> inline std::string request<intgemm8avx2>()        { return "intgemm8avx2";    }
template <> inline std::string request<intgemm8avx512>()      { return "intgemm8avx512";  }
template <> inline std::string request<intgemm8avx512vnni>()  { return "intgemm8avx512vnni";  }
template <> inline std::string request<intgemm16>()           { return "intgemm16";       }
template <> inline std::string request<intgemm16sse2>()       { return "intgemm16sse2";   }
template <> inline std::string request<intgemm16avx2>()       { return "intgemm16avx2";   }
template <> inline std::string request<intgemm16avx512>()     { return "intgemm16avx512"; }
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

  if(str == "packed16")
    return Type::packed16;
  if(str == "packed8avx2")
    return Type::packed8avx2;
  if(str == "packed8avx512")
    return Type::packed8avx512;

  if(str == "intgemm8")
    return Type::intgemm8;
  if(str == "intgemm8ssse3")
    return Type::intgemm8ssse3;
  if(str == "intgemm8avx2")
    return Type::intgemm8avx2;
  if(str == "intgemm8avx512")
    return Type::intgemm8avx512;
  if(str == "intgemm8avx512vnni")
    return Type::intgemm8avx512vnni;

  if(str == "intgemm16")
    return Type::intgemm16;
  if(str == "intgemm16sse2")
    return Type::intgemm16sse2;
  if(str == "intgemm16avx2")
    return Type::intgemm16avx2;
  if(str == "intgemm16avx512")
    return Type::intgemm16avx512;

  ABORT("Unknown type {}", str);
}

template <typename T>
inline Type typeId();

template <> inline Type typeId<int8_t>()   { return Type::int8;  }
template <> inline Type typeId<int16_t>()  { return Type::int16; }
template <> inline Type typeId<int32_t>()  { return Type::int32; }
template <> inline Type typeId<int64_t>()  { return Type::int64; }

template <> inline Type typeId<uint8_t>()  { return Type::uint8;  }
template <> inline Type typeId<uint16_t>() { return Type::uint16; }
template <> inline Type typeId<uint32_t>() { return Type::uint32; }
template <> inline Type typeId<uint64_t>() { return Type::uint64; }

template <> inline Type typeId<float16>()  { return Type::float16; }
template <> inline Type typeId<float>()    { return Type::float32; }
template <> inline Type typeId<double>()   { return Type::float64; }

template <> inline Type typeId<packed16>()      { return Type::packed16;      }
template <> inline Type typeId<packed8avx2>()   { return Type::packed8avx2;   }
template <> inline Type typeId<packed8avx512>() { return Type::packed8avx512; }

template <> inline Type typeId<intgemm8>()            { return Type::intgemm8;            }
template <> inline Type typeId<intgemm8ssse3>()       { return Type::intgemm8ssse3;       }
template <> inline Type typeId<intgemm8avx2>()        { return Type::intgemm8avx2;        }
template <> inline Type typeId<intgemm8avx512>()      { return Type::intgemm8avx512;      }
template <> inline Type typeId<intgemm8avx512vnni>()  { return Type::intgemm8avx512vnni;  }
template <> inline Type typeId<intgemm16>()           { return Type::intgemm16;           }
template <> inline Type typeId<intgemm16sse2>()       { return Type::intgemm16sse2;       }
template <> inline Type typeId<intgemm16avx2>()       { return Type::intgemm16avx2;       }
template <> inline Type typeId<intgemm16avx512>()     { return Type::intgemm16avx512;     }


// Abort if given C++ does not correspond to runtime type
template <typename T>
void matchOrAbort(Type type) {
  ABORT_IF(!matchType<T>(type),
           "Requested type ({}) and underlying type ({}) do not match",
           request<T>(),
           type);
}

namespace typeFitting { // own namespace instead of in class, otherwise we get error "explicit specialization in non-namespace scope"

  // Helper function for fitsIntoMax() below
  // Returns the 'capacity' of a type: number of digits for integers,
  // max_exponent for floats. We ignore the mantissa for floats.
  template<typename X> constexpr int capacity() {
    static_assert(std::is_arithmetic<X>::value || std::is_same<X,HalfFloat>::value,
                  "Wrong type for this template");
    return (std::is_integral<X>::value
            ? std::numeric_limits<X>::digits
            : std::numeric_limits<X>::max_exponent);
 }


  // Compare max for different types as constexpr, so can be used at compile-time to determine if RequestType type max fits into ReturnType max, see std::conditional below.
  template <typename RequestType, typename ReturnType>
  constexpr bool fitsIntoMax() {
    // We can't just compare std::numeric_limits<>::max(), because Clang-10
    // complains about rounding errors when implicitly converting int to float
    return ((!std::is_integral<RequestType>::value // RequestType is a float
             && std::is_integral<ReturnType>::value) // ReturnType an integer
            ? capacity<RequestType>() < capacity<ReturnType>() // special case
            : capacity<RequestType>() <= capacity<ReturnType>()); // normal case
  } // for built-in types everything is constexpr

}

template <typename ReturnType>
class NumericLimits {
private:

  template <typename MaxType> void setLimitsMax() {
    max    = (ReturnType)std::numeric_limits<MaxType>::max();
    min    = (ReturnType)std::numeric_limits<MaxType>::min();
    lowest = (ReturnType)std::numeric_limits<MaxType>::lowest();
  }

  template <typename RequestType>
  void setLimits() {
    // check if the maximum of type RequestType fits into ReturnType
    constexpr bool fits = typeFitting::fitsIntoMax<RequestType, ReturnType>();
    // sanity check:
    static_assert(fits || typeFitting::fitsIntoMax<ReturnType, RequestType>(),
                  "RequestType doesn't fit into ReturnType, and ReturnType doesn't "
                  "fit into RequestType. fitsIntoMax is broken!");
    // and then use the smaller of each types to determine max, min, lowest.
    using MaxType = typename std::conditional<fits, RequestType, ReturnType>::type;
    setLimitsMax<MaxType>();
    // @TODO: should we rather abort if the RequestType does not fit into ReturnType instead of clipping to smaller type?
    // ABORT_IF(!fits, "Type {} is too small to contain max of type {}", typeId<ReturnType>(), typeId<RequestType>());
  }

  void setLimits(Type type) {
    DISPATCH_BY_TYPE0(type, setLimits);
  }

public:
  ReturnType max;
  ReturnType min;
  ReturnType lowest;

  NumericLimits(Type type) {
    setLimits(type);
  }
};

}  // namespace marian

// custom specialization of std::hash can be injected in namespace std
namespace std {
  template<> struct hash<::marian::Type> {
    size_t operator()(const ::marian::Type& type) const noexcept {
      return (size_t)type; // type is already a unique value of type size_t
    }
  };
}
