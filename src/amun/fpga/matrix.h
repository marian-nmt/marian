#pragma once
#include "common/base_matrix.h"
#include "types-fpga.h"
#include "array.h"

namespace amunmt {
namespace FPGA {
namespace mblas {

class Matrix : public BaseMatrix {
public:
  Matrix(const OpenCLInfo &openCLInfo);
  Matrix(const OpenCLInfo &openCLInfo, size_t rows, size_t cols, bool zero = false);
  Matrix(const OpenCLInfo &openCLInfo, size_t rows, size_t cols, float *val);

  Matrix(const Matrix &other);
  Matrix(Matrix &&other);

  virtual ~Matrix();

  virtual size_t size() const
  { return arr_.size(); }

  const uint &sizeUInt() const
  { return arr_.sizeUInt(); }

  virtual size_t dim(size_t i) const
  { return dims_[i]; }

  const uint &dimUInt(size_t i) const
  { return dims_[i]; }

  virtual void Resize(size_t rows, size_t cols, size_t beam = 1, size_t batches = 1);

  void Reshape(size_t rows, size_t cols, size_t beam, size_t batches);

  void Reshape2D();

  const OpenCLInfo &GetOpenCLInfo() const
  { return arr_.GetOpenCLInfo(); }

  const cl_mem &data() const
  { return arr_.data(); }

  virtual std::string Debug(size_t verbosity = 1) const;

  void Swap(Matrix &other);

  void Set(const float *data);

protected:
  uint dims_[SHAPE_SIZE];
  Array<float> arr_;

};


}
}
}

