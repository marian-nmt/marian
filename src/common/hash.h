#pragma once

#include <functional>

namespace marian {
namespace util {

template <class T> using hash = std::hash<T>;

// This combinator is based on boost::hash_combine, but uses
// std::hash as the hash implementation. Used as a drop-in
// replacement for boost::hash_combine.
template <class T, class HashType = std::size_t>
inline void hash_combine(HashType& seed, T const& v) {
  hash<T> hasher;
  seed ^= static_cast<HashType>(hasher(v)) + 0x9e3779b9 + (seed<<6) + (seed>>2);
}

// Hash a whole chunk of memory, mostly used for diagnostics
template <class T, class HashType = std::size_t>
inline HashType hashMem(const T* beg, size_t len, HashType seed = 0) {
  for(auto it = beg; it < beg + len; ++it)
    hash_combine(seed, *it);
  return seed;
}

}
}
