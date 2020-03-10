#include "common/fastopt.h"

#include <utility>

namespace marian {

const std::unique_ptr<const FastOpt> FastOpt::uniqueNullPtr{nullptr};

// see fastopt.h for comments
namespace fastopt_helpers {

// helper structs for dynamic type conversion and specializations
// for different conversion scenarios.

// general template, mostly for numerical and logical types
template <typename To, typename From>
struct Convert { 
  static inline To apply(const From& from) { 
    return (To)from; 
  } 
};

// specialization for translating from string, @TODO check if this is required at all, mostly for compilation now.
template <typename To>
struct Convert<To, std::string> { 
  static inline To apply(const std::string& /* from */) { 
    ABORT("Not implemented");
  }
};

// convert anything to string, checked at compile-time
template <typename From>
struct Convert<std::string, From> { 
  static inline std::string apply(const From& from) { 
    return std::to_string(from);
  }
};

// do nothing conversion for std::string
template <>
struct Convert<std::string, std::string> { 
  static inline std::string apply(const std::string& from) { 
    return from;
  }
};

// helper class for FastOpt::as<T>() used for specializations
template <typename T>
T As<T>::apply(const FastOpt& node) {
  ABORT_IF(!node.isScalar(), "Node is not a scalar node");

  if(node.isBool())
    return Convert<T, bool>::apply(node.value_.as<bool>());
  else if(node.isInt())
    return Convert<T, int64_t>::apply(node.value_.as<int64_t>());
  else if(node.isFloat())
    return Convert<T, double>::apply(node.value_.as<double>());
  else if(node.isString())
    return Convert<T, std::string>::apply(node.value_.as<std::string>());      
  else {
    ABORT("Casting of value failed");
  }
}

// specializations for simple types
template struct As<bool>;
template struct As<int>;
template struct As<unsigned long>;
template struct As<float>;
template struct As<double>;
template struct As<std::string>;

// specialization of above class for std::vector<T>
template <typename T>
std::vector<T> As<std::vector<T>>::apply(const FastOpt& node) {
  ABORT_IF(!node.isSequence(), "Node is not a sequence node");

  std::vector<T> seq;
  for(const auto& elem : node.array_)
    seq.push_back(elem->as<T>());
  return seq;
}

// specializations for simple vector types
template struct As<std::vector<bool>>;
template struct As<std::vector<int>>;
// Windows, Linux based OS and Mac have different type definitions for 'unsigned long'.
// So, we need an explicit definitions for uint64_t, that cover different platforms.
// Otherwise, there's a linking error on windows or Linux or Mac.
// https://software.intel.com/en-us/articles/size-of-long-integer-type-on-different-architecture-and-os/
// https://stackoverflow.com/questions/32021860/c-should-you-size-t-with-a-regular-array
// MacOS: size_t = unsigned long (8 bytes), uint64_t = unsigned long long (8 bytes)
// Linux: size_t = unsigned long (8 bytes), uint64_t = unsigned long (8 bytes)
// Windows: size_t = unsigned long long (8 bytes), uint64_t = unsigned long long (8 bytes)
template struct As<std::vector<unsigned long long>>;
template struct As<std::vector<unsigned long>>;
template struct As<std::vector<float>>;
template struct As<std::vector<double>>;
template struct As<std::vector<std::string>>;

// specialization of above class for std::pair<T>
template <typename T1, typename T2>
std::pair<T1, T2> As<std::pair<T1, T2>>::apply(const FastOpt& node) {
  ABORT_IF(!node.isSequence(), "Node is not a sequence node");
  ABORT_IF(node.size() != 2, "Sequence must contain two elements in order to convert to pair");
  return std::make_pair(node[0].as<T1>(), node[1].as<T2>());
}

template struct As<std::pair<int, int>>;

}
}
