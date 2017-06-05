#include <cuda.h>
#include <limits>

#include "data/types.h"
#include "tensors/tensor.h"
#include "translator/helpers.h"

namespace marian {

__global__ void gSetColumn(float* d_in,
                           size_t n_columns,
                           size_t n_rows,
                           size_t noColumn,
                           float value) {
  size_t rowNumber = threadIdx.x + blockDim.x * blockIdx.x;
  size_t index = noColumn + rowNumber * n_columns;

  if(index < n_columns * n_rows) {
    d_in[index] = value;
  }
}

void SetColumn(Tensor in, size_t col, float value) {
  int nRows = in->shape()[0] * in->shape()[2] * in->shape()[3];
  int nColumns = in->shape()[1];

  int nBlocks = nRows / 512 + ((nRows % 512 == 0) ? 0 : 1);
  int nThreads = std::min(512, nRows);

  gSetColumn<<<nBlocks, nThreads>>>(in->data(), nColumns, nRows, col, value);
}

void suppressUnk(Expr probs) {
  SetColumn(probs->val(), UNK_ID, std::numeric_limits<float>::lowest());
}

void suppressWord(Expr probs, Word id) {
  SetColumn(probs->val(), id, std::numeric_limits<float>::lowest());
}
}
