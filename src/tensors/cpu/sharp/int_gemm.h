#pragma once

#include "tensors/tensor.h"

namespace marian {
namespace cpu {
namespace int16 {

const int BITS = 10;

void Quantize16(marian::Tensor out,
                const marian::Tensor in,
                float /*clipValue*/);

void Quantize8(marian::Tensor out,
               const marian::Tensor in,
               float clipValue);

// This operates on floats after processing so doesn't care about int8_t vs
// int16_t.
void AddBias(marian::Tensor C, const marian::Tensor Bias);

void ProdInt16(marian::Tensor C,
               const marian::Tensor A,
               const marian::Tensor B,
               float scale);

void ProdInt8(marian::Tensor C,
              const marian::Tensor A,
              const marian::Tensor B,
              float scale,
              float clipValue);

}  // namespace int16
}  // namespace cpu
}  // namespace marian
