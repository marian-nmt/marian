#include <algorithm>
#include <random>

#include "tensors/cpu/backend.h"
#include "tensors/tensor_operators.h"

namespace marian {
namespace cpu {

void Dropout(Tensor tensor, float dropProb) {
  auto cpuBackend
      = std::static_pointer_cast<cpu::Backend>(tensor->getBackend());
  auto &gen = cpuBackend->getRandomGenerator();
  std::bernoulli_distribution dist(1.f - dropProb);
  std::generate(tensor->data(), tensor->data() + tensor->size(), [&]() {
    return (float)dist(gen) / (1.f - dropProb);
  });
}
}  // namespace cpu
}  // namespace marian
