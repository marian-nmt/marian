#include "gru.h"
#include "simd_math_prims.h"

void gElementwiseOps(float* out,
                    const float* state,
                    const float* ru,
                    const float* h,
                    const float* t1,
                    const float* t2,
                    const float* b,
                    const float* bx1,
                    const float* bx2,
                    size_t rows, size_t cols) {
  for(int j = 0; j < rows; ++j) {
    float* rowOut = out + j * cols;
    const float* rowRu = ru + j * cols * 2;
    const float* rowT1 = t1 + j * cols * 2;
    
    const float* rowH = h + j * cols;
    const float* rowT2 = t2 + j * cols;
    const float* rowState = state + j * cols;
    
    for(int i = 0; i < cols; ++i) {
      float ev1 = expapprox(-(rowRu[i] + b[i] + rowT1[i]));
      float r = 1.0 / (1.0 + ev1);
      
      int k = i + cols;
      float ev2 = expapprox(-(rowRu[k] + b[k] + rowT1[k]));
      float u = 1.0 / (1.0 + ev2);              

      float hv = rowH[i] + bx1[i];
      float t2v = rowT2[i] + bx2[i];
      hv = tanhapprox(hv + r * t2v);
      rowOut[i] = (1.0 - u) * hv + u * rowState[i];
    }
  }
}
