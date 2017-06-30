#pragma once
#include "cnpy/cnpy.h"
#include "matrix.h"

namespace amunmt {
namespace FPGA {

class NpzConverter {
private:
  class NpyMatrixWrapper {
  public:
    NpyMatrixWrapper(const cnpy::NpyArray& npy)
    : npy_(npy) {}

    size_t size() const {
      return size1() * size2();
    }

    float* data() const {
      return (float*)npy_.data;
    }

    size_t size1() const {
      return npy_.shape[0];
    }

    size_t size2() const {
      if(npy_.shape.size() == 1)
        return 1;
      else
        return npy_.shape[1];
    }

  private:
    const cnpy::NpyArray& npy_;

  };

public:
  NpzConverter(const std::string& file);

  ~NpzConverter() {
      model_.destruct();
  }

  mblas::Matrix GetMatrix(
      const OpenCLInfo &openCLInfo,
		  const std::string& key,
		  bool transpose = false) const;

private:
  cnpy::npz_t model_;

};

}
}

