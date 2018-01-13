#pragma once

#include "training/sparse_tensor.h"

namespace marian {

class GradientDropBase {
  float* residual;
  float* velocity;
  float* temp_d;
  float cut_off;
  int step;
  int _device;

  void grad_drop_do(float* grads,
                    float* residual,
                    float* velocity,
                    float* tmp,
                    int len,
                    float rate,
                    float m);

public:
  void dropGraph(Tensor t, SparseTensor destination, double rate = 0.99, double momentum = 0.0);
};

typedef Ptr<GradientDropBase> GradientDrop;
}
