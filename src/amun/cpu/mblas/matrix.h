#pragma once

#include <cmath>
#include <iostream>
#include <vector>
#include <sstream>

#include <blaze/Math.h>
#include "phoenix_functions.h"
#include "common/base_matrix.h"
#include "common/exception.h"

namespace amunmt {
namespace CPU {

namespace mblas {

typedef blaze::DynamicVector<float, blaze::rowVector> Vector;
typedef blaze::DynamicVector<float, blaze::columnVector> ColumnVector;

//////////////////////////////////////////////////////////////////////////////////////////////
class Matrix : public BaseMatrix, public blaze::DynamicMatrix<float, blaze::rowMajor>
{
public:
  typedef blaze::DynamicMatrix<float, blaze::rowMajor> Parent;

  Matrix()
    : Parent()
  {}

  Matrix(size_t rows, size_t cols)
    : Parent(rows, cols)
  {}

  template<typename T>
  Parent& operator=(const T &other) {
    return Parent::operator=(other);
  }

  virtual size_t dim(size_t i) const
  {
  	switch (i) {
  	case 0: return Parent::rows();
  	case 1: return Parent::columns();
  	case 2: return 1;
  	case 3: return 1;
  	default:
  		abort();
  	}
  }

  virtual void Resize(size_t rows, size_t cols, size_t beam = 1, size_t batches = 1)
  {
    amunmt_UTIL_THROW2("Not implemented");
  }

};


//////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, bool SO = blaze::rowMajor>
class BlazeMatrix : public BaseMatrix, public blaze::CustomMatrix<T, blaze::unaligned,
                                             blaze::unpadded,
                                             blaze::rowMajor> {
  public:
    typedef T value_type;
    typedef typename std::vector<value_type>::iterator iterator;
    typedef typename std::vector<value_type>::const_iterator const_iterator;
    typedef blaze::CustomMatrix<value_type,
                                blaze::unaligned,
                                blaze::unpadded,
                                SO> BlazeBase;

    BlazeMatrix() {}

    BlazeMatrix(size_t rows, size_t columns, value_type val = 0)
     : data_(rows * columns, val) {
       BlazeBase temp(data_.data(), rows, columns);
       std::swap(temp, *(BlazeBase*)this);
    }

    template <class MT>
    BlazeMatrix(const MT& rhs)
     : data_(rhs.rows() * rhs.columns()) {
       BlazeBase temp(data_.data(), rhs.rows(), rhs.columns());
       temp = rhs;
       std::swap(temp, *(BlazeBase*)this);
    }

    virtual size_t dim(size_t i) const
    {
    	switch (i) {
    	case 0: return BlazeBase::rows();
    	case 1: return BlazeBase::columns();
    	case 2: return 1;
    	case 3: return 1;
    	default:
    		abort();
    	}
    }

    virtual void Resize(size_t rows, size_t columns, size_t beam = 1, size_t batches = 1)
    {
      assert(beam == 1);
      assert(batches == 1);
      data_.resize(rows * columns);
      BlazeBase temp(data_.data(), rows, columns);
      std::swap(temp, *(BlazeBase*)this);
    }

    BlazeMatrix<T, SO>& operator=(const value_type& val) {
      *(BlazeBase*)this = val;
      return *this;
    }

    template <class MT>
    BlazeMatrix<T, SO>& operator=(const MT& rhs) {
      Resize(rhs.rows(), rhs.columns());
      BlazeBase temp(data_.data(), rhs.rows(), rhs.columns());
      temp = rhs;
      std::swap(temp, *(BlazeBase*)this);
      return *this;
    }

    operator BlazeBase&() {
      return *(BlazeBase*)this;
    }

    iterator begin() {
      return data_.begin();
    }

    iterator end() {
      return data_.end();
    }

    const_iterator begin() const{
      return data_.begin();
    }

    const_iterator end() const {
      return data_.end();
    }

    size_t size() const {
      return data_.size();
    }

    void swap(BlazeMatrix<T, SO>& rhs) {
      std::swap(data_, rhs.data_);
      std::swap(static_cast<BlazeBase&>(*this), static_cast<BlazeBase&>(rhs));
    }

  private:
    std::vector<value_type> data_;
};

////////////////////////////////////////////////////////////////////////
class ArrayMatrix : public BlazeMatrix<float, blaze::rowMajor>
{
	typedef BlazeMatrix<float, blaze::rowMajor> Parent;
  public:
    ArrayMatrix()
      :Parent()
    {}

    ArrayMatrix(size_t rows, size_t columns, value_type val = 0)
      : Parent(rows, columns, val)
    {}

    template <class MT>
    ArrayMatrix(const MT& rhs)
      : Parent(rhs)
    {}

};

////////////////////////////////////////////////////////////////////////
template <class M>
std::string Debug(const M& m)
{
  std::stringstream strm;
  strm << m.rows() << "x" << m.columns() << ":"; // ":\n";
  for (size_t row = 0; row < m.rows(); ++row) {
	  float rowSum = 0;
	  for (size_t col = 0; col < m.columns(); ++col) {
		  //strm << m(row, col) << " ";
		  rowSum += m(row, col);
	  }
	  //strm << std::endl;
	  strm << rowSum << " ";
  }
  return strm.str();
}

template <bool byRow, class MT, class VT>
MT& AddBiasVector(MT& m, const VT& b) {
  if(byRow) {
    for(size_t i = 0; i < m.rows(); ++i)
      // @TODO: replace this with row vector
      blaze::row(m, i) += blaze::row(b, 0);
  }
  else {
    for(size_t i = 0; i < m.columns(); ++i)
      // @TODO: replace this with row vector
      blaze::column(m, i) += blaze::column(b, 0);
  }
  return m;
}

//Matrix& Swap(Matrix& Out, Matrix& In);

template <class MT>
void Reshape(MT& m, size_t rows, size_t cols) {
  assert(rows * cols == m.rows() * m.columns());
  MT temp(rows, cols);
  for(size_t i = 0; i < m.rows(); ++i) {
    for(size_t j = 0; j < m.columns(); ++j) {
      size_t k = i * m.columns() + j;
      size_t i2 = k / cols;
      size_t j2 = k % cols;
      temp(i2, j2) = m(i, j);
    }
  }
  temp.swap(m);
}

template <bool byRow, class MT, class MT1>
MT Mean(const MT1& in) {
  MT out;
  if(byRow) {
    size_t rows = in.rows();
    size_t cols = in.columns();
    out.resize(1, cols);
    blaze::row(out, 0) = blaze::row(in, 0);
    for(size_t i = 1; i < rows; ++i)
      blaze::row(out, 0) += blaze::row(in, i);
    out *= 1.0f / rows;
  }
  else {
    size_t rows = in.rows();
    size_t cols = in.columns();
    out.resize(rows, 1);
    blaze::column(out, 0) = blaze::column(in, 0);
    for(size_t i = 1; i < cols; ++i)
      blaze::column(out, 0) += blaze::column(in, i);
    out *= 1.0f / cols;
  }
  return std::move(out);
}

typedef std::pair<size_t, size_t> RowPair;
typedef std::vector<RowPair> RowPairs;
typedef std::vector<RowPair> DeviceRowPairs;

const bool byRow = true;
const bool byColumn = false;

template <bool byRow, class MT, class MT1, class MT2>
MT Concat(const MT1& m1, const MT2& m2) {
  MT out = m1;
  if(byRow) {
    assert(m1.columns() == m2.columns());
    size_t rows1 = m1.rows();
    size_t rows2 = m2.rows();
    size_t rows = rows1 + rows2;
    size_t cols = m1.columns();
    out.resize(rows, cols);
    for(size_t i = 0; i < rows2; ++i)
      blaze::row(out, rows1 + i) = blaze::row(m2, i);
  }
  else {
    assert(m1.rows() == m2.rows());
    size_t cols1 = m1.columns();
    size_t cols2 = m2.columns();
    size_t cols = cols1 + cols2;
    size_t rows = m1.rows();
    out.resize(rows, cols);
    for(size_t i = 0; i < cols2; ++i)
      blaze::column(out, cols1 + i) = blaze::column(m2, i);
  }
  return std::move(out);
}

template <bool byRow, class MT, class MT1>
MT Assemble(const MT1& in,
            const std::vector<uint>& indices) {
  MT out;
  if(byRow) {
    size_t rows = indices.size();
    size_t cols = in.columns();
    out.resize(rows, cols);
    for(size_t i = 0; i < rows; ++i)
      blaze::row(out, i) = blaze::row(in, indices[i]);
  }
  else {
    size_t rows = in.rows();
    size_t cols = indices.size();
    out.resize(rows, cols);
    for(size_t i = 0; i < cols; ++i)
      blaze::column(out, i) = blaze::column(in, indices[i]);
  }
  return std::move(out);
}

template <class MT>
void SafeSoftmax(MT& Out) {
  size_t rows = Out.rows();
  size_t cols = Out.columns();
  float sum[rows];
  for (int j = 0; j < rows; ++j) {
    float maxRowValue = 0.0f;
    for (int i = 0; i < cols; ++i) {
      maxRowValue = std::max(maxRowValue, Out(j,i));
    }
    sum[j] = 0;
    for (int i = 0; i < cols; ++i) {
      Out(j, i) = expapprox(Out(j, i) - maxRowValue);
      sum[j] += Out(j, i);
    }
    for(int i = 0; i < cols; ++i) {
      Out(j, i) /= sum[j];
    }
  }
}

template <class MT>
void LogSoftmax(MT& Out) {
  size_t rows = Out.rows();
  size_t cols = Out.columns();
  float sum[rows];
  for (int j = 0; j < rows; ++j) {
    sum[j] = 0;
    for (int i = 0; i < cols; ++i) {
      sum[j] += expapprox(Out(j, i));
    }
    for(int i = 0; i < cols; ++i) {
      Out(j, i) -= logapprox(sum[j]);
    }
  }
}

template <class MT>
void Softmax(MT& Out) {
  size_t rows = Out.rows();
  size_t cols = Out.columns();
  float sum[rows];
  for (int j = 0; j < rows; ++j) {
    float maxRowValue = 0.0f;
    for (int i = 0; i < cols; ++i) {
      maxRowValue = std::max(maxRowValue, Out(j,i));
    }

    sum[j] = 0;
    for (int i = 0; i < cols; ++i) {
      Out(j,i) = expapprox(Out(j, i) - maxRowValue);
      sum[j] += Out(j, i);
    }

    for(int i = 0; i < cols; ++i) {
      Out(j, i) /= sum[j];
    }
  }
}

template <class MT, class Functor, class MT1, class MT2>
MT Broadcast(const Functor& functor, const MT1& m1, const MT2& m2) {
  size_t rows1 = m1.rows();
  size_t rows2 = m2.rows();

  size_t rows = rows1 * rows2;
  size_t cols = m1.columns();

  MT out(rows, cols);
  for (int j = 0; j < rows; ++j) {
    size_t r1 = j % rows1;
    size_t r2 = j / rows1;

    blaze::row(out, j) =
      blaze::forEach(blaze::row(m1, r1) + blaze::row(m2, r2),
                     functor);
  }
  return std::move(out);
}

template<class MT>
void LayerNormalization(MT& in, const MT& gamma, const MT& beta, float eps=1e-5f) {
  eps=1e-5f;
  // std::cerr << "LAYER NORM" << std::endl;
  // std::cerr << std::endl;
  size_t rows = in.rows();
  size_t cols = in.columns();

  for (int j = 0; j < rows; ++j) {
    // std::cerr << "PRE ";
    // for (int i = 0; i < 10; ++i) {
      // std::cerr << in(j, i) << " ";
    // }
    // std::cerr << std::endl;
    //
    float sum = 0.0f;
    for (int i = 0; i < cols; ++i) {
      sum += in(j, i);
    }

    float mean = sum / cols;

    float sigma = 0.0f;
    for (int i = 0; i < cols; ++i) {
      sigma += (in(j, i) - mean) * (in(j, i) - mean);
    }
    sigma /= cols;

    sigma = sqrt(sigma + eps);

    // std::cerr << "MIDD ";
    // for (int i = 0; i < 10; ++i) {
      // std::cerr << ( (in(j, i) - mean) / sigma) << " ";
    // }
    // std::cerr << std::endl;

    for (int i = 0; i < cols; ++i) {
      in(j, i) = gamma(i, 0) * ( (in(j, i) - mean) / sigma) + beta(i, 0);
    }

    // std::cerr << "POST ";
    // for (int i = 0; i < 10; ++i) {
      // std::cerr << in(j, i) << " ";
    // }
    // std::cerr << std::endl;
  }
  // std::cerr << "LAYER NORM: DONE" << std::endl;
}

template<class MT>
void LayerNormalization(MT& in, const MT& gamma, float eps=1e-9) {
  size_t rows = in.rows();
  size_t cols = in.columns();

  for (int j = 0; j < rows; ++j) {
    float sum = 0.0f;
    for (int i = 0; i < cols; ++i) {
      sum += in(j, i);
    }

    float mean = sum / cols;

    float sigma = 0.0f;
    for (int i = 0; i < cols; ++i) {
      sigma += (in(j, i) - mean) * (in(j, i) - mean);
    }
    sigma /= cols;

    sigma = sqrt(sigma + eps);

    for (int i = 0; i < cols; ++i) {
      in(j, i) = gamma(i, 0) * ( (in(j, i) - mean) / sigma);
    }
  }
}

}
}
}
