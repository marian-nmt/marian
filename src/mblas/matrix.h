#pragma once

#include <cmath>
#include <iostream>
#include <vector>

#include <blaze/Math.h>
#include "phoenix_functions.h"

namespace mblas {

namespace bpp = boost::phoenix::placeholders;

template <typename T, bool SO = blaze::rowMajor>
class BlazeMatrix : public blaze::CustomMatrix<T, blaze::unaligned,
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

    void resize(size_t rows, size_t columns) {
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
      resize(rhs.rows(), rhs.columns());
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
    
    size_t Rows() const {
      return BlazeBase::rows();
    }

    size_t Cols() const {
      return BlazeBase::columns();
    }

    void Clear() {
      BlazeBase temp;
      std::swap(temp, *(BlazeBase*)this);
      data_.clear();
    }
    
    void Resize(size_t r, size_t c) {
      resize(r, c);
    }
  
    void Resize(size_t r, size_t c, value_type val) {
      resize(r, c);
      std::fill(data_.begin(), data_.end(), val);
    }
    
    void Reshape(size_t r, size_t c) {
      assert(r * c == size());
      resize(r, c);
    }
    
    std::vector<value_type>& GetVec() {
      return data_;
    }
    
    const std::vector<value_type>& GetVec() const {
      return data_;
    }
    
    void swap(BlazeMatrix<T, SO>& rhs) {
      std::swap(data_, rhs.data_);
      std::swap(static_cast<BlazeBase&>(*this), static_cast<BlazeBase&>(rhs));
    }
  
  private:
    std::vector<value_type> data_;                                       
};

typedef BlazeMatrix<float, blaze::rowMajor> Matrix;

template <class M>
void Debug(const M& m, size_t maxRows = 5, size_t maxCols = 5) {
  std::cerr << m.Rows() << " " << m.Cols() << std::endl;
  for(size_t i = 0; i < m.Rows() && i < maxRows; ++i) {
    for(size_t j = 0; j < m.Cols() && j < maxCols; ++j) {
      std::cerr << m.GetVec()[i * m.Cols() + j] << " ";
    }
    std::cerr << std::endl;
  }
}

template <class M>
void Debug2(const M& m) {
  std::cerr << m.Rows() << " " << m.Cols() << std::endl;
  for(size_t i = 0; i < m.Rows(); ++i) {
    for(size_t j = 0; j < m.Cols(); ++j) {
      std::cerr << m(i, j) << " ";
    }
    std::cerr << std::endl;
  }
}

Matrix& Swap(Matrix& Out, Matrix& In);

Matrix& Mean(Matrix& Out, const Matrix& In);

Matrix& Transpose(Matrix& Out, const Matrix& In);

Matrix& Transpose(Matrix& Out);

Matrix& Copy(Matrix& Out, const Matrix& In);

Matrix& PasteRow(Matrix& Out,
                 const Matrix& In,
                 const size_t r = 0, const size_t c = 0);

Matrix& CopyRow(Matrix& Out,
                const Matrix& In,
                const size_t r = 0, const size_t c = 0);

typedef std::pair<size_t, size_t> RowPair;
typedef std::vector<RowPair> RowPairs;
typedef std::vector<RowPair> DeviceRowPairs;

Matrix& Concat(Matrix& Out, const Matrix& In);

Matrix& CopyRows(Matrix& Out,
                 const Matrix& In,
                 const RowPair* devPairs,
                 size_t numPairs);

Matrix& CopyRows(Matrix& Out,
                 const Matrix& In,
                 const RowPairs& pairs);

Matrix& Assemble(Matrix& Out,
                 const Matrix& In,
                 const std::vector<size_t>& indeces);

Matrix& AssembleCols(Matrix& Out,
                 const Matrix& In,
                 const std::vector<size_t>& indeces);

Matrix& Slice(Matrix& Out,
              const Matrix& In,
              size_t n, size_t dim);

Matrix& Prod(Matrix& C, const Matrix& A, const Matrix& B,
             bool transA = false, bool transB = false);

Matrix& Softmax(Matrix& Out);
Matrix& SoftmaxLog(Matrix& Out);

template <class Functor>
Matrix& Broadcast(Functor functor, Matrix& Out, const Matrix& In) {
  size_t rows1 = Out.Rows();
  size_t rows2 = In.Rows();

  size_t rows = rows1 * rows2;
  size_t cols  = Out.Cols();

  Matrix Temp(rows, cols, 1.0);

  float* d_out = Temp.data();
  const float* d_in1 = Out.data();
  const float* d_in2 = In.data();

  for(int j = 0; j < rows; ++j) {
    float* rowOut = d_out + j * cols;
    const float* rowIn1 = d_in1 + (j % rows1) * cols;
    const float* rowIn2 = d_in2 + (j / rows1) * cols;
    
    for(int i = 0; i < cols; ++i)
      rowOut[i] = functor(rowIn1[i], rowIn2[i]);
  }
  
  Swap(Out, Temp);
  return Out;
}

template <class Functor>
Matrix& BroadcastColumn(Functor functor, Matrix& Out, const Matrix& In) {
  // @TODO: Make this efficient with special kernel!
  Matrix InTemp;
  Transpose(InTemp, In);

  Transpose(Out);
  Broadcast(functor, Out, InTemp);
  Transpose(Out);
  return Out;
}

template <class Functor>
Matrix& BroadcastVecColumn(Functor functor, Matrix& Out, const Matrix& In) {
  size_t rows  = Out.Rows();
  size_t cols = Out.Cols();

  float* d_out = Out.data();
  const float* d_in = In.data();

  for(int j = 0; j < cols; ++j) {    
    for(int i = 0; i < rows; ++i) {
      float* rowOut = d_out + i * cols + j;
      const float* rowIn  = d_in + i;
      *rowOut = functor(*rowOut, *rowIn);      
    }
  }
  return Out;
}

template <class Functor>
Matrix& BroadcastVec(Functor functor, Matrix& Out, const Matrix& In) {
  size_t rows  = Out.Rows();
  size_t cols = Out.Cols();

  float* d_out = Out.data();
  const float* d_in = In.data();

  for(int j = 0; j < rows; ++j) {
    float* rowOut = d_out + j * cols;
    for(int i = 0; i < cols; ++i)
      rowOut[i] = functor(rowOut[i], d_in[i]);
  }
  
  return Out;
}

template <class Functor>
Matrix& Element(Functor functor, Matrix& Out) {
  float* d_out = Out.data();
  for(int i = 0; i < Out.size(); ++i)
    d_out[i] = functor(d_out[i]);
  return Out;
}

template <class Functor>
Matrix& Element(Functor functor,
                Matrix& Out, const Matrix& In) {
  float* d_out = Out.data();
  const float* d_in = In.data();

  for(int i = 0; i < Out.size(); ++i)
    d_out[i] = functor(d_out[i], d_in[i]);

  return Out;
}

template <class Functor>
Matrix& Element(Functor functor,
                Matrix& Out, const Matrix& In1, const Matrix& In2) {
  
  float* d_out = Out.data();
  const float* d_in1 = In1.data();
  const float* d_in2 = In2.data();
  
  for(int i = 0; i < Out.size(); ++i)
    d_out[i] = functor(d_out[i], d_in1[i], d_in2[i]);

  return Out;
}

}
