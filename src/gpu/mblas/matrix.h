#pragma once

#include <memory>
#include <sstream>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>

#include "common/base_matrix.h"

#include "gpu/types-gpu.h"

namespace amunmt {
namespace GPU {
namespace mblas {

using namespace thrust::placeholders;


template <typename T>
class TMatrix : public BaseMatrix {
  public:
    typedef DeviceVector<T> VecType;
    typedef typename VecType::value_type value_type;
    typedef typename VecType::iterator iterator;
    typedef typename VecType::const_iterator const_iterator;

    TMatrix()
    : rows_(0)
    , cols_(0)
    , data_(nullptr)
    {
    }

    TMatrix(size_t rows, size_t cols)
    : rows_(rows)
    , cols_(cols)
    , data_(new VecType(size()))
    {
    }

    TMatrix(size_t rows, size_t cols, value_type val)
    : rows_(rows)
    , cols_(cols)
    , data_(new VecType(size(), val))
    {
    }

    TMatrix(TMatrix&& m)
    : TMatrix()
    {
      swap(m);
    }

    TMatrix(const TMatrix& m)
    : rows_(m.rows_)
    , cols_(m.cols_)
    , data_(new VecType(*m.data_))
    {
    }

    value_type operator()(size_t i, size_t j) const {
      return (*data_)[i * cols_ + j];
    }

    ~TMatrix()
    {
      Clear();
    }

    void Set(size_t i, size_t j, float value)  {
      (*data_)[i * cols_ + j] = value;
    }

    size_t Rows() const {
      return rows_;
    }

    size_t Cols() const {
      return cols_;
    }

    void Resize(size_t rows, size_t cols) {
       abort();
    }

    void ResizeOrig(size_t rows, size_t cols) {
      if (cols * rows > size()) {
        if (data_) {
          data_->resize(rows * cols);
        }
        else {
          data_ = new VecType(rows * cols);
        }
      }
      rows_ = rows;
      cols_ = cols;
    }

    void ResizeNew(size_t rows, size_t cols) {
      if (data_) {
        if ((cols*rows) > data_->size()) {
          //HANDLE_ERROR(cudaStreamSynchronize(0));
          VecType *newData = new VecType(rows * cols);
          //HANDLE_ERROR(cudaStreamSynchronize(0));

          thrust::copy(data_->begin(), data_->begin() + size(), newData->begin());
          //HANDLE_ERROR(cudaStreamSynchronize(0));

          delete data_;
          data_ = newData;
        }
      }
      else {
        data_ = new VecType(rows * cols);
        //HANDLE_ERROR(cudaStreamSynchronize(0));
      }
      rows_ = rows;
      cols_ = cols;
    }

    void Reshape(size_t rows, size_t cols) {
      rows_ = rows;
      cols_ = cols;
    }

    virtual std::string Debug() const
    {
      std::stringstream strm;
      strm << Rows() << "x" << Cols() << " " << data_ << ":";
      /*
      for (size_t row = 0; row < Rows(); ++row) {
        float rowSum = 0;
        for (size_t col = 0; col < Cols(); ++col) {
          rowSum += (*this)(row, col);
        }
        strm << rowSum << " ";
      }
      */
      return strm.str();
    }

    void Clear() {
      delete data_;
      data_ = nullptr;
      rows_ = 0;
      cols_ = 0;
    }

    value_type* data() {
      return thrust::raw_pointer_cast(data_->data());
    }

    const value_type* data() const {
      return thrust::raw_pointer_cast(data_->data());
    }

    iterator begin() {
      return data_->begin();
    }

    iterator end() {
      return data_->begin() + size();
      // return data_.end();
    }

    const_iterator begin() const{
      return data_->begin();
    }

    const_iterator end() const {
      return data_->begin() + size();
      // return data_.end();
    }

    size_t size() const {
      // return data_.size();
      return cols_ * rows_;
    }

    void swap(TMatrix &other)
    {
      std::swap(rows_, other.rows_);
      std::swap(cols_, other.cols_);
      std::swap(data_, other.data_);
    }

  private:
    size_t rows_;
    size_t cols_;
    VecType *data_;
};

typedef TMatrix<float> Matrix;
typedef TMatrix<int> IMatrix;


}  // namespace mblas
}  // namespace GPU
}
