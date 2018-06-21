#pragma once

#include <memory>
#include <sstream>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>

#include "common/exception.h"
#include "common/base_tensor.h"
#include "gpu/types-gpu.h"
#include "handles.h"
#include "vector.h"

namespace amunmt {
namespace GPU {
namespace mblas {

using namespace thrust::placeholders;

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
class TTensor : public BaseTensor {
  public:
    typedef T value_type;

    TTensor()
    {
      dim_[0] = 0;
      dim_[1] = 0;
      dim_[2] = 0;
      dim_[3] = 0;
    }

    TTensor(unsigned rows, unsigned cols, unsigned c, unsigned d, bool zero = false)
    {
      dim_[0] = rows;
      dim_[1] = cols;
      dim_[2] = c;
      dim_[3] = d;

      unsigned newSize = rows * cols * c * d;
      vec_.newSize(newSize);

      if (zero) {
        HANDLE_ERROR( cudaMemsetAsync(vec_.data(), 0, newSize * sizeof(T), CudaStreamHandler::GetStream()) );
      }
    }

    TTensor(TTensor&& m)
    : TTensor()
    {
      swap(m);
    }

    TTensor(const TTensor& m)
    : vec_(m.vec_)
    {
      dim_[0] = m.dim_[0];
      dim_[1] = m.dim_[1];
      dim_[2] = m.dim_[2];
      dim_[3] = m.dim_[3];
    }

    ~TTensor()
    {
    }

    virtual unsigned size() const
    {
      return vec_.size();
    }

    virtual unsigned dim(unsigned i) const
    {
      return dim_[i];
    }

    void Resize(unsigned rows, unsigned cols, unsigned c = 1, unsigned d = 1) {
      unsigned newSize = cols * rows * c * d;
      vec_.resize(newSize);

      dim_[0] = rows;
      dim_[1] = cols;
      dim_[2] = c;
      dim_[3] = d;
    }

    void NewSize(unsigned rows, unsigned cols, unsigned c = 1, unsigned d = 1) {
      unsigned newSize = cols * rows * c * d;
      vec_.newSize(newSize);

      dim_[0] = rows;
      dim_[1] = cols;
      dim_[2] = c;
      dim_[3] = d;
    }

    virtual std::string Debug(unsigned verbosity = 1) const
    {
      std::stringstream strm;
      strm << BaseTensor::Debug(verbosity) << " ";

      if (verbosity == 1) {
        if (dim(1) > 1) {
          HANDLE_ERROR( cudaStreamSynchronize(CudaStreamHandler::GetStream()));

          unsigned maxCol = std::min((unsigned) 2, dim(1));

          T tmp[2];
          HANDLE_ERROR( cudaMemcpy(tmp, vec_.data(), maxCol * sizeof(T), cudaMemcpyDeviceToHost) );

          for (size_t i = 0; i < maxCol; ++i) {
            strm << " " << tmp[i];
          }

          if (dim(1) > 3) {
            strm << "...";
            HANDLE_ERROR( cudaMemcpy(tmp, vec_.data() + dim(1) - maxCol, maxCol * sizeof(T), cudaMemcpyDeviceToHost) );
            for (size_t i = 0; i < maxCol; ++i) {
              strm << tmp[i] << " ";
            }
          }

          if (dim(0) > 1 || dim(2) > 1 || dim(3) > 1) {
            // last row
            strm << "/";


            HANDLE_ERROR( cudaMemcpy(tmp, vec_.data() + size() - dim(1), maxCol * sizeof(T), cudaMemcpyDeviceToHost) );
            for (unsigned i = 0; i < maxCol; i++) {
              strm << " " << tmp[i];
            }

            if (dim(1) > 3) {
              HANDLE_ERROR( cudaMemcpy(tmp, vec_.data() + size() - maxCol, maxCol * sizeof(T), cudaMemcpyDeviceToHost) );

              strm << "...";
              for (unsigned i = 0; i < maxCol; ++i) {
                strm << tmp[i] << " ";
              }
            }

          }
        }
      }
      else if (verbosity == 2) {
        HANDLE_ERROR( cudaStreamSynchronize(CudaStreamHandler::GetStream()));

        T h_data[size()];

        HANDLE_ERROR( cudaMemcpy(
            &h_data,
            vec_.data(),
            size() * sizeof(T),
            cudaMemcpyDeviceToHost) );

        for (unsigned i = 0; i < size(); ++i) {
          strm << " " << h_data[i];
        }
      }

      return strm.str();
    }

    value_type* data() {
      return vec_.data();
    }

    const value_type* data() const {
      return vec_.data();
    }

    void swap(TTensor &other)
    {
      std::swap(dim_, other.dim_);
      vec_.swap(other.vec_);
    }

  private:
    unsigned dim_[SHAPE_SIZE];
    Vector<T> vec_;
};

typedef TTensor<float> Tensor;


}  // namespace mblas
}  // namespace GPU
}
