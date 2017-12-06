#pragma once

#include "training/sparse_tensor.h"

namespace marian {

class GradientDropBase {
  float* feedback;
  float* temp_d;
  float cut_off;
  int step;
  int _device;

  void grad_drop_do(float* data,
                    float* errors,
                    float* tmp,
                    int len,
                    float rate);

public:
  void dropGraph(Tensor t, SparseTensor destination, double rate = 0.99);
};

typedef Ptr<GradientDropBase> GradientDrop;
}
