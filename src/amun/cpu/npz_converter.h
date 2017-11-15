#pragma once

#include "cnpy/cnpy.h"
#include "mblas/matrix.h"

namespace amunmt {
namespace CPU {

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

    typedef blaze::CustomMatrix<float, blaze::unaligned,
      blaze::unpadded, blaze::rowMajor> BlazeWrapper;

    bool has(std::string key) const {
      auto it = model_.find(key);
      return (it != model_.end());
    }


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

    mblas::Matrix operator[](const std::string& key) const {
      BlazeWrapper matrix;
      auto it = model_.find(key);
      if(it != model_.end()) {
        NpyMatrixWrapper np(it->second);
        matrix = BlazeWrapper(np.data(), np.size1(), np.size2());
      }
      else {
        if (key.find("gamma") == std::string::npos) {
          std::cerr << "Missing " << key << std::endl;
        }
      }

      mblas::Matrix ret;
      ret = matrix;
      return std::move(ret);
    }

    mblas::Matrix getFirstOfMany(const std::vector<std::pair<std::string, bool>> keys) const {
      BlazeWrapper matrix;
      for (auto key : keys) {
        auto it = model_.find(key.first);
        if(it != model_.end()) {
          NpyMatrixWrapper np(it->second);
          matrix = BlazeWrapper(np.data(), np.size1(), np.size2());
          mblas::Matrix ret;
          if (key.second) {
            const auto matrix2 = blaze::trans(matrix);
            ret = matrix2;
          } else {
            ret = matrix;
          }
          return std::move(ret);
        }
      }
      std::cerr << "Matrix not found: " << keys[0].first << "\n";

      mblas::Matrix ret;
      return std::move(ret);
    }

    mblas::Matrix operator()(const std::string& key,
                                   bool transpose) const {
      BlazeWrapper matrix;
      auto it = model_.find(key);
      if(it != model_.end()) {
        NpyMatrixWrapper np(it->second);
        matrix = BlazeWrapper(np.data(), np.size1(), np.size2());
      } else {
          std::cerr << "Missing " << key << std::endl;
      }
      mblas::Matrix ret;
      if (transpose) {
        const auto matrix2 = blaze::trans(matrix);
        ret = matrix2;
      } else {
        ret = matrix;
      }
      return std::move(ret);
    }

  private:
    cnpy::npz_t model_;
    bool destructed_;
};

}
}

