#pragma once

#include <functional>

namespace marian {
namespace util {

template <class T> using hash = std::hash<T>;

template <class T>
inline void hash_combine(std::size_t& seed, T const& v) {
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
}

}
}