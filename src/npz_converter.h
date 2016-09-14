#pragma once

#include "cnpy/cnpy.h"
#include "tensor.h"

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

    void Load(const std::string& key, marian::Tensor& tensor) const {
      auto it = model_.find(key);
      if(it != model_.end()) {
        NpyMatrixWrapper np(it->second);
        tensor.allocate({(int)np.size1(), (int)np.size2()});
        std::vector<float> data(np.data(), np.data() + np.size());
        tensor.Load(data);
      }
      else {
        std::cerr << "Missing " << key << std::endl;
      }
    }

  private:
    cnpy::npz_t model_;
    bool destructed_;
};
