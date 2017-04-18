#pragma once

#include <memory>
#include <sstream>

#include "common/base_matrix.h"

#include "gpu/types-gpu.h"

namespace amunmt {
namespace GPU {
namespace mblas {

using namespace thrust::placeholders;

template<class IteratorT1, class IteratorT2>
void copy_n(IteratorT1 inBegin, size_t size, IteratorT2 outBegin);

void GetValues(const float*  in, int* d_indices, float* scores, int n);

template <class VecType>
class TMatrix : public BaseMatrix {
  public:
    typedef typename VecType::value_type value_type;
    typedef typename VecType::iterator iterator;
    typedef typename VecType::const_iterator const_iterator;

    TMatrix()
    : rows_(0), cols_(0)
    {}

    TMatrix(size_t rows, size_t cols)
    : rows_(rows), cols_(cols), data_(rows_ * cols_)
    {}

    TMatrix(size_t rows, size_t cols, value_type val)
    : rows_(rows), cols_(cols), data_(rows_ * cols_, val)
    {}

    TMatrix(TMatrix&& m)
    : rows_(m.rows_), cols_(m.cols_), data_(std::move(m.data_)) {}

    TMatrix(const TMatrix& m) = delete;

    float GetValue(int i, int j) const {
      return data_[i * cols_ + j];
    }

    virtual std::vector<float> GetScores(const std::vector<std::pair<int, int>>& indices) {
      if (d_ids.size() < indices.size()) {
        d_ids.resize(indices.size());
        h_ids.resize(indices.size());
        d_scores.resize(indices.size());
      }
      h_scores.resize(indices.size());


      for (size_t i = 0; i < indices.size(); ++i) {
        h_ids[i] = indices[i].first * Cols() + indices[i].second;
      }
      copy_n(h_ids.begin(), indices.size(), d_ids.begin());


      GetValues(data(),
                thrust::raw_pointer_cast(d_ids.data()),
                thrust::raw_pointer_cast(d_scores.data()),
                (int)indices.size());
      copy_n(d_scores.begin(), indices.size(), h_scores.begin());

      return h_scores;

    }

    value_type operator()(size_t i, size_t j) const {
      return data_[i * cols_ + j];
    }

    void Set(size_t i, size_t j, float value)  {
      data_[i * cols_ + j] = value;
    }

    size_t Rows() const {
      return rows_;
    }

    size_t Cols() const {
      return cols_;
    }

    void Resize(size_t rows, size_t cols) {
      if (cols * rows > data_.size()) {
        data_.resize(rows * cols);
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
      strm << Rows() << "x" << Cols() << ": ";
      for (size_t row = 0; row < Rows(); ++row) {
        float rowSum = 0;
        for (size_t col = 0; col < Cols(); ++col) {
          rowSum += (*this)(row, col);
        }
        strm << rowSum << " ";
      }
      return strm.str();
    }

    void Purge() {
      Clear();
      VecType temp;
      data_.swap(temp);
    }

    void Clear() {
      data_.clear();
      rows_ = 0;
      cols_ = 0;
    }

    VecType& GetVec() {
      return data_;
    }

    const VecType& GetVec() const {
      return data_;
    }

    value_type* data() {
      return thrust::raw_pointer_cast(data_.data());
    }

    const value_type* data() const {
      return thrust::raw_pointer_cast(data_.data());
    }

    iterator begin() {
      return data_.begin();
    }

    iterator end() {
      return data_.begin() + size();
      // return data_.end();
    }

    const_iterator begin() const{
      return data_.begin();
    }

    const_iterator end() const {
      return data_.begin() + size();
    }

    size_t size() const {
      return cols_ * rows_;
    }


    operator bool() const {
      return (int)size() != 0;
    }

  private:
    size_t rows_;
    size_t cols_;
    VecType data_;
    thrust::device_vector<int> d_ids;
    thrust::host_vector<int> h_ids;
    std::vector<float> h_scores;
    thrust::device_vector<float> d_scores;
};

typedef TMatrix<DeviceVector<float>> Matrix;
typedef TMatrix<DeviceVector<int>> IMatrix;


}  // namespace mblas
}  // namespace GPU
}
