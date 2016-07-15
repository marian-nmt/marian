#pragma once

#include <cmath>
#include <iostream>
#include <vector>

#define EIGEN_DONT_PARALLELIZE
#include <eigen3/Eigen/Dense>

#include "cblas.h"
#include "phoenix_functions.h"

namespace mblas {

typedef Eigen::Matrix<float,
                      Eigen::Dynamic,
                      Eigen::Dynamic> Matrix;

typedef Eigen::Matrix<float,
                      Eigen::Dynamic,
                      Eigen::Dynamic,
                      Eigen::RowMajor> RMatrix;

typedef Eigen::Matrix<float, 1, Eigen::Dynamic> Vector;
typedef Eigen::Matrix<float, Eigen::Dynamic, 1> RVector;

typedef Eigen::Map<Matrix> MatrixMap;

//Matrix operator*(const Matrix& m1, const Matrix& m2);
                      
template <class M>
void Debug(const M& m, size_t pos = 0, size_t l = 5) {
  std::cerr << m.rows() << " " << m.cols() << std::endl;
  for(size_t i = 0; i < m.rows(); ++i) {
    for(size_t j = pos; j < m.cols() && j < pos + l; ++j) {
      std::cerr << m(i, j) << " ";
    }
    std::cerr << std::endl;
    if(i == 4)
      break;
  }
}

template <class M>
void Debug2(const M& m) {
  std::cerr << m.rows() << " " << m.cols() << std::endl;
  for(size_t i = 0; i < m.rows(); ++i) {
    for(size_t j = 0; j < m.cols(); ++j) {
      std::cerr << m(i, j) << " ";
    }
    std::cerr << std::endl;
  }
}

typedef std::pair<size_t, size_t> RowPair;
typedef std::vector<RowPair> RowPairs;
typedef std::vector<RowPair> DeviceRowPairs;

template <class M>
M& Assemble(M& Out, const M& In,
            const std::vector<size_t>& indeces) {
  RowPairs rowPairs;
  for(size_t i = 0; i < indeces.size(); i++)
    rowPairs.emplace_back(i, indeces[i]);
  Out.resize(rowPairs.size(), In.cols());
  
  for(int j = 0; j < rowPairs.size(); ++j) {
    size_t dstId = rowPairs[j].first;
    size_t srcId = rowPairs[j].second;
    Out.row(dstId) = In.row(srcId);
  }
  
  return Out;
}

Matrix& Slice(Matrix& Out,
              const Matrix& In,
              size_t n, size_t dim);

Matrix& Softmax(Matrix& Out);
Matrix& SoftmaxLog(Matrix& Out);

template <class Functor, class M1, class M2>
Matrix& Broadcast(Functor functor, M1& Out, const M2& In) {
  size_t rows1 = Out.rows();
  size_t rows2 = In.rows();

  size_t rows = rows1 * rows2;
  size_t cols  = Out.cols();

  Matrix Temp(rows, cols);
  
  #pragma omp for schedule(dynamic, 10)
  for(size_t j = 0; j < cols; ++j) {
    const float* colOut = Out.data() + j * rows1;
    const float* colIn = In.data() + j * rows2;
    float* colT = Temp.data() + j * rows;
    
    for(size_t i = 0; i < rows; i++) {
      size_t r1 = i % rows1;
      size_t r2 = i / rows1;
      colT[i] = functor(colOut[r1], colIn[r2]);
    }
  }
  
  Out.swap(Temp);
  return Out;
}

template <class Functor>
Matrix& BroadcastColumn(Functor functor, Matrix& Out, const Matrix& In) {
  // @TODO: Make this efficient
  Matrix InTemp =  In.transpose();
  Matrix OutTemp = Out.transpose();
  Broadcast(functor, OutTemp, InTemp);
  Out = OutTemp.transpose();
  return Out;
}

template <class Functor>
Matrix& BroadcastVecColumn(Functor functor, Matrix& Out, const Matrix& In) {
  size_t rows  = Out.rows();
  size_t cols = Out.cols();

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
  size_t rows = Out.rows();
  size_t cols = Out.cols();

  float* d_out = Out.data();
  const float* d_in = In.data();

  for(int j = 0; j < cols; ++j) {
    float* colOut = d_out + j * rows;
    for(int i = 0; i < rows; ++i)
      colOut[i] = functor(colOut[i], d_in[j]);
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
