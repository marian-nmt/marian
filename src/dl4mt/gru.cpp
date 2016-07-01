#include "gru.h"
#include "simd_math_prims.h"

void gElementwiseOps(float* out,
                    const float* state,
                    const float* ruh,
                    const float* t,
                    const float* b,
                    const float* bx1,
                    const float* bx2,
                    size_t rows, size_t cols) {
  for(int j = 0; j < rows; ++j) {
    float* rowOut = out + j * cols;
    const float* rowRuh = ruh + j * cols * 3;
    const float* rowT = t + j * cols * 3;
    
    const float* rowH = rowRuh + 2 * cols;
    const float* rowT2 = rowT + 2 * cols;
    const float* rowState = state + j * cols;
    
    for(int i = 0; i < cols; ++i) {
      float ev1 = expapprox(-(rowRuh[i] + b[i] + rowT[i]));
      float r = 1.0 / (1.0 + ev1);
      
      int k = i + cols;
      float ev2 = expapprox(-(rowRuh[k] + b[k] + rowT[k]));
      float u = 1.0 / (1.0 + ev2);              

      float hv = rowH[i] + bx1[i];
      float t2v = rowT2[i] + bx2[i];
      hv = tanhapprox(hv + r * t2v);
      rowOut[i] = (1.0 - u) * hv + u * rowState[i];
    }
  }
}
