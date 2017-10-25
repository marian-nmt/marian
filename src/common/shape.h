#pragma once

#include <cstdint>
#include <iostream>
#include <string>

// #include "exception.h"

namespace marian {

/**
 * @brief Represents the size of each dimension in a tensor.
 *
 * Note: this class currently is hard-coded to four dimensions.
 */

class Shape {
  private:
    std::vector<int> shape_;
    std::vector<int> stride_;
    std::vector<int> bstride_;

  public:
    Shape() : shape_{1}, stride_{1}, bstride_{0} {}

    Shape(std::initializer_list<int> il) : Shape() {
      shape_.resize(il.size());
      std::copy(il.begin(), il.end(), begin());
      updateStrides();
    }

    void updateStrides() {
      stride_.resize(shape_.size());
      bstride_.resize(shape_.size());

      stride_.back() = 1;
      bstride_.back() = shape_.back() == 1 ? 0 : stride_.back();

      for(int i = size() - 2; i >= 0; --i) {
        stride_[i] = stride_[i + 1] * shape_[i + 1];
        bstride_[i] = shape_[i] == 1 ? 0 : stride_[i];
      }
    }

    Shape(const Shape& shape) : Shape() {
      shape_.resize(shape.size());
      std::copy(shape.begin(), shape.end(), begin());
      updateStrides();
    }

    inline void set(int i, int val) {
      dim(i) = val;
      updateStrides();
    }

    inline int& dim(int i) {
      if(i < 0)
        return shape_[shape_.size() + i];
      else
        return shape_[i];
    }

    inline const int& dim(int i) const { return const_cast<Shape&>(*this).dim(i); }

    inline int operator[](int i) { return dim(i); }

    inline int operator[](int i) const { return dim(i); }

    inline int& back() { return shape_.back(); }

    inline int stride(int i) const {
      return stride_[i];
    }

    inline int bstride(int i) const {
      return bstride_[i];
    }

    inline size_t size() const { return shape_.size(); }

    inline int elements() const {
      int el = 1;
      for(auto s : shape_)
        el *= s;
      return el;
    }

    inline int index(const std::vector<int>& d) const {
      int i = 0;
      for(int j = 0; j < shape_.size(); ++j)
        i += d[j] * stride_[j];
      return i;
    }

    inline int bindex(const std::vector<int>& d) const {
      int i = 0;
      for(int j = 0; j < shape_.size(); ++j)
        i += d[j] * bstride_[j];
      return i;
    }

    inline void dims(int i, std::vector<int>& d) const {
      d.resize(shape_.size());
      for(int j = 0; j < d.size(); ++j)
        d[j] = (i / stride_[j]) % shape_[j];
    }

    auto begin() -> decltype(shape_.begin()) { return shape_.begin(); }
    auto begin() const -> decltype(shape_.begin()) { return shape_.begin(); }

    auto end() -> decltype(shape_.end()) { return shape_.end(); }
    auto end() const -> decltype(shape_.end()) { return shape_.end(); }

    bool operator==(const Shape& other) const {
      return size() == other.size() && std::equal(begin(), end(), other.begin());
    }

    bool operator!=(const Shape& other) const { return !(*this == other); }

    friend std::ostream& operator<<(std::ostream& strm, const Shape& shape) {
      strm << "shape=" << shape[0];
      for(int i = 1; i < shape.size(); ++i)
        strm << "x" << shape[i];
      strm << " size=" << shape.elements() << " ("
           << shape.elements() * sizeof(float) << "B)";
      return strm;
    }
};

}
