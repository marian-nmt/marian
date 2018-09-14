#pragma once

#include <functional>

namespace marian {
namespace util {

template <class T> using hash = std::hash<T>;

// This combinator is based on boost::hash_combine, but uses
// std::hash as the hash implementation. Used as a drop-in
// replacement for boost::hash_combine.

template <class T>
inline void hash_combine(std::size_t& seed, T const& v) {
    hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
}

}
}