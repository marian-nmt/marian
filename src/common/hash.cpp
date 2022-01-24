#include <string>

#include "hash.h"
#include "common/shape.h"

namespace std {
size_t hash<pair<string, marian::Shape>>::operator()(pair<string, marian::Shape> const& k) const {
  size_t seed = hash<string>{}(k.first);
  marian::util::hash_combine(seed, k.second.hash());
  return seed;
}
}  // namespace std
