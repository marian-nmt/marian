#ifndef ARRAY2D_H_
#define ARRAY2D_H_

#include <iostream>
#include <algorithm>
#include <cassert>
#include <vector>
#include <string>

template<typename T>
class Array2D {
 public:
  typedef typename std::vector<T>::reference reference;
  typedef typename std::vector<T>::const_reference const_reference;
  typedef typename std::vector<T>::iterator iterator;
  typedef typename std::vector<T>::const_iterator const_iterator;
  Array2D() : width_(0), height_(0) {}
  Array2D(unsigned w, unsigned h, const T& d = T()) :
    width_(w), height_(h), data_(w*h, d) {}
  Array2D(const Array2D& rhs) :
    width_(rhs.width_), height_(rhs.height_), data_(rhs.data_) {}
  bool empty() const { return data_.empty(); }
  void resize(unsigned w, unsigned h, const T& d = T()) {
    data_.resize(w * h, d);
    width_ = w;
    height_ = h;
  }
  const Array2D& operator=(const Array2D& rhs) {
    data_ = rhs.data_;
    width_ = rhs.width_;
    height_ = rhs.height_;
    return *this;
  }
  void fill(const T& v) { data_.assign(data_.size(), v); }
  unsigned width() const { return width_; }
  unsigned height() const { return height_; }
  reference operator()(unsigned i, unsigned j) {
    return data_[offset(i, j)];
  }
  void clear() { data_.clear(); width_=0; height_=0; }
  const_reference operator()(unsigned i, unsigned j) const {
    return data_[offset(i, j)];
  }
  iterator begin_col(unsigned j) {
    return data_.begin() + offset(0,j);
  }
  const_iterator begin_col(unsigned j) const {
    return data_.begin() + offset(0,j);
  }
  iterator end_col(unsigned j) {
    return data_.begin() + offset(0,j) + width_;
  }
  const_iterator end_col(unsigned j) const {
    return data_.begin() + offset(0,j) + width_;
  }
  iterator end() { return data_.end(); }
  const_iterator end() const { return data_.end(); }
  const Array2D<T>& operator*=(const T& x) {
    std::transform(data_.begin(), data_.end(), data_.begin(),
        std::bind2nd(std::multiplies<T>(), x));
  }
  const Array2D<T>& operator/=(const T& x) {
    std::transform(data_.begin(), data_.end(), data_.begin(),
        std::bind2nd(std::divides<T>(), x));
  }
  const Array2D<T>& operator+=(const Array2D<T>& m) {
    std::transform(m.data_.begin(), m.data_.end(), data_.begin(), data_.begin(), std::plus<T>());
  }
  const Array2D<T>& operator-=(const Array2D<T>& m) {
    std::transform(m.data_.begin(), m.data_.end(), data_.begin(), data_.begin(), std::minus<T>());
  }

 private:
  inline unsigned offset(unsigned i, unsigned j) const {
    assert(i<width_);
    assert(j<height_);
    return i + j * width_;
  }

  unsigned width_;
  unsigned height_;

  std::vector<T> data_;
};

template <typename T>
Array2D<T> operator*(const Array2D<T>& l, const T& scalar) {
  Array2D<T> res(l);
  res *= scalar;
  return res;
}

template <typename T>
Array2D<T> operator*(const T& scalar, const Array2D<T>& l) {
  Array2D<T> res(l);
  res *= scalar;
  return res;
}

template <typename T>
Array2D<T> operator/(const Array2D<T>& l, const T& scalar) {
  Array2D<T> res(l);
  res /= scalar;
  return res;
}

template <typename T>
Array2D<T> operator+(const Array2D<T>& l, const Array2D<T>& r) {
  Array2D<T> res(l);
  res += r;
  return res;
}

template <typename T>
Array2D<T> operator-(const Array2D<T>& l, const Array2D<T>& r) {
  Array2D<T> res(l);
  res -= r;
  return res;
}

template <typename T>
inline std::ostream& operator<<(std::ostream& os, const Array2D<T>& m) {
  for (unsigned i=0; i<m.width(); ++i) {
    for (unsigned j=0; j<m.height(); ++j)
      os << '\t' << m(i,j);
    os << '\n';
  }
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const Array2D<bool>& m) {
  os << ' ';
  for (unsigned j=0; j<m.height(); ++j)
    os << (j%10);
  os << "\n";
  for (unsigned i=0; i<m.width(); ++i) {
    os << (i%10);
    for (unsigned j=0; j<m.height(); ++j)
      os << (m(i,j) ? '*' : '.');
    os << (i%10) << "\n";
  }
  os << ' ';
  for (unsigned j=0; j<m.height(); ++j)
    os << (j%10);
  os << "\n";
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const Array2D<std::vector<bool> >& m) {
  os << ' ';
  for (unsigned j=0; j<m.height(); ++j)
    os << (j%10) << "\t";
  os << "\n";
  for (unsigned i=0; i<m.width(); ++i) {
    os << (i%10);
    for (unsigned j=0; j<m.height(); ++j) {
      const std::vector<bool>& ar = m(i,j);
      for (unsigned k=0; k<ar.size(); ++k)
        os << (ar[k] ? '*' : '.');
    }
    os << "\t";
    os << (i%10) << "\n";
  }
  os << ' ';
  for (unsigned j=0; j<m.height(); ++j)
    os << (j%10) << "\t";
  os << "\n";
  return os;
}

#endif

