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
    NpzConverter(const std::string& file);
    ~NpzConverter();

    void Destruct();

    std::shared_ptr<mblas::Matrix> get(const std::string& key, bool mandatory, bool transpose = false) const;
    std::shared_ptr<mblas::Matrix> getFirstOfMany(const std::vector<std::pair<std::string, bool>> keys, bool mandatory) const;

  private:
    cnpy::npz_t model_;
    bool destructed_;
};

}
}


