#pragma once

#include <functional>

namespace marian {
namespace util {

template <class T> using hash = std::hash<T>;

/**
 * Combine hash values.
 * This combinator is based on boost::hash_combine, but uses std::hash as the hash implementation.
 * Used as a drop-in replacement for boost::hash_combine.
 */
template <class T, class HashType = std::size_t>
inline void hash_combine(HashType& seed, T const& v) {
  hash<T> hasher;
  seed ^= static_cast<HashType>(hasher(v)) + 0x9e3779b9 + (seed<<6) + (seed>>2);
}

/** Hash a whole chunk of memory. */
template <class T, class HashType = std::size_t>
inline HashType hashMem(const T* beg, size_t len) {
  HashType seed = 0;
  for(auto it = beg; it < beg + len; ++it)
    hash_combine(seed, *it);
  return seed;
}

}  // namespace util

struct Shape;  // Forward declaration
}  // namespace marian

namespace std {
/**
 * std::hash specialization for the string-shape pair used as a cache key in transformer.h.
 */
template <>
struct hash<pair<string, marian::Shape>> {
  size_t operator()(pair<string, marian::Shape> const& k) const;
};
}
