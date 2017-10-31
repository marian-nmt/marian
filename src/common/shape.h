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

struct Shape {
  public:
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

    void resize(size_t n) {
      shape_.resize(n, 1);
      updateStrides();
    }

    const int* data() const {
      return shape_.data();
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
      if(i >= 0)
        return shape_[i];
      else
        return shape_[size() + i];
    }

    inline const int& dim(int i) const { return const_cast<Shape&>(*this).dim(i); }

    inline int operator[](int i) { return dim(i); }

    inline int operator[](int i) const { return dim(i); }

    inline int& back() { return shape_.back(); }

    inline int stride(int i) const {
      if(i >= 0)
        return stride_[i];
      else
        return stride_[size() + i];
    }

    inline int bstride(int i) const {
      if(i >= 0)
        return bstride_[i];
      else
        return bstride_[size() + i];
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

    int axis(int ax) {
      if(ax < 0)
        return size() + ax;
      else
        return ax;
    }

    static Shape broadcast(const std::vector<Shape>& shapes) {
      int maxDims = 0;
      for(auto& s : shapes)
        if(s.size() > maxDims)
          maxDims = s.size();

      Shape shape;
      shape.resize(maxDims);

      for(auto& s : shapes) {
        for(int i = 0; i < s.size(); ++i) {
          ABORT_IF(shape[-i] != s[-i] && shape[-i] != 1 && s[-i] != 1,
                   "Shapes cannot be broadcasted");
          shape.set(-i, std::max(shape[-i], s[-i]));
        }
      }
      return shape;
    }

    template <typename T>
    static Shape broadcast(const std::initializer_list<T>& il) {
      return broadcast(std::vector<T>(il));
    }

    template <typename T>
    static Shape broadcast(const std::vector<T>& nodes) {
      int maxDims = 0;
      for(auto& n : nodes)
        if(n->shape().size() > maxDims)
          maxDims = n->shape().size();

      Shape shape;
      shape.resize(maxDims);

      for(auto& node : nodes) {
        Shape shapen = node->shape();
        for(int i = 1; i <= shapen.size(); ++i) {
          ABORT_IF(shape[-i] != shapen[-i] && shape[-i] != 1 && shapen[-i] != 1,
                   "Shapes cannot be broadcasted");
          shape.set(-i, std::max(shape[-i], shapen[-i]));
        }
      }
      return shape;
    }
};

}
