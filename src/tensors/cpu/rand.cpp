#include <algorithm>
#include <random>

#include "tensors/cpu/backend.h"
#include "tensors/tensor_operators.h"

namespace marian {
namespace cpu {

void Uniform(Tensor tensor, float a, float b) {
  auto cpuBackend
      = std::static_pointer_cast<cpu::Backend>(tensor->getBackend());
  auto &engine = cpuBackend->getRandomGenerator();
  std::uniform_real_distribution<float> dist(a, b);
  auto gen = std::bind(dist, engine);

  auto begin = tensor->data<float>();
  auto end   = tensor->data<float>() + tensor->size();
  std::generate(begin, end, gen);
}

void Normal(Tensor tensor, float mju, float sigma) {
  auto cpuBackend
      = std::static_pointer_cast<cpu::Backend>(tensor->getBackend());
  auto &engine = cpuBackend->getRandomGenerator();
  std::normal_distribution<float> dist(mju, sigma);
  auto gen = std::bind(dist, engine);

  auto begin = tensor->data<float>();
  auto end   = tensor->data<float>() + tensor->size();
  std::generate(begin, end, gen);
}

}  // namespace cpu
}  // namespace marian
