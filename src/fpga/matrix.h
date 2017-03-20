#pragma once
#include "common/base_matrix.h"
#include "types.h"

namespace amunmt {
namespace FPGA {
namespace mblas {

class Matrix : public BaseMatrix {
public:
  Matrix(const cl_context &context);
  Matrix(const cl_context &context, size_t rows, size_t cols, float val);
  Matrix(const cl_context &context, size_t rows, size_t cols, float *val);

  virtual size_t dim(size_t i) const
  {
    switch (i) {
    case 0: return rows_;
    case 1: return cols_;
    case 2: return 1;
    case 3: return 1;
    default:
      abort();
    }
  }

  virtual void Resize(size_t rows, size_t cols, size_t beam = 1, size_t batches = 1);

  virtual std::string Debug(bool detailed = false) const;

protected:
  const cl_context &context_;
  cl_mem mem_;
  float *data_;
  size_t rows_, cols_;
};


}
}
}

