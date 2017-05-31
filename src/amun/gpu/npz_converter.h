#pragma once

#include "cnpy/cnpy.h"
#include "mblas/matrix_functions.h"

namespace amunmt {
namespace GPU {

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

        float operator()(size_t i, size_t j) const {
          return ((float*)npy_.data)[i * size2() + j];
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
    NpzConverter(const std::string& file)
      : model_(cnpy::npz_load(file)),
        destructed_(false) {
      }

    ~NpzConverter() {
      if(!destructed_)
        model_.destruct();
    }

    void Destruct() {
      model_.destruct();
      destructed_ = true;
    }

    mblas::Matrix get(const std::string& key, bool transpose = false) const {
      mblas::Matrix matrix;
      auto it = model_.find(key);
      if(it != model_.end()) {
        NpyMatrixWrapper np(it->second);
        matrix.Resize(np.size1(), np.size2());
        mblas::copy(np.data(), np.size(), matrix.data(), cudaMemcpyHostToDevice);
      }
      else {
        std::cerr << "Missing " << key << std::endl;
      }

      if (transpose) {
    	  mblas::Transpose(matrix);
      }

      //std::cerr << "key=" << key << " " << matrix.Debug(1) << std::endl;
      return std::move(matrix);
    }

    std::shared_ptr<mblas::Matrix> getPtr(const std::string& key, bool transpose = false) const {

      std::shared_ptr<mblas::Matrix> ret;
      auto it = model_.find(key);
      if(it != model_.end()) {
        NpyMatrixWrapper np(it->second);
        mblas::Matrix *matrix = new mblas::Matrix(np.size1(), np.size2(), 1, 1);
        mblas::copy(np.data(), np.size(), matrix->data(), cudaMemcpyHostToDevice);

        if (transpose) {
      	  mblas::Transpose(*matrix);
        }

        ret.reset(matrix);
      }
      else {
        std::cerr << "Missing " << key << std::endl;
      }


      //std::cerr << "key=" << key << " " << matrix.Debug(1) << std::endl;
      return ret;
    }

  private:
    cnpy::npz_t model_;
    bool destructed_;
};

}
}


