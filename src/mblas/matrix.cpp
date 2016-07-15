#include "matrix.h"
#include "simd_math_prims.h"

namespace mblas {

void gSlice(float* out, const float* in,
            size_t n, size_t dim,
            size_t rows, size_t cols) {
  for(int j = 0; j < rows; j++) {
    float* rowOut = out + j * dim;
    const float* rowIn = in + j * cols + n * dim;

    for(int i = 0; i < dim; ++i)
        rowOut[i] = rowIn[i];
  }
}

//Matrix operator*(const Matrix& m1, const Matrix& m2) {
//  Matrix out(m1.rows(), m2.cols());
//  Prod(out, m1, m2);
//  return std::move(out);
//}

Matrix& Slice(Matrix& Out,
              const Matrix& In,
              size_t n, size_t dim) {

  Out.resize(In.rows(), dim);

  float* d_out = Out.data();
  const float* d_in = In.data();

  gSlice(d_out, d_in, n, dim, In.rows(), In.cols());
  return Out;
}

Matrix& Softmax(Matrix& Out) {
  float sum[Out.rows()];
  for(int j = 0; j < Out.rows(); ++j) {
    sum[j] = 0;
    for(int i = 0; i < Out.cols(); ++i) {
      Out.row(j)[i] = expapprox(Out.row(j)[i]);
      sum[j] += Out.row(j)[i];
    }
    for(int i = 0; i < Out.cols(); ++i) {
      Out.row(j)[i] /= sum[j];
    }
  }
  return Out;
}

Matrix& SoftmaxLog(Matrix& Out) {
  float sum[Out.cols()];
  for(int j = 0; j < Out.cols(); ++j) {
    sum[j] = 0;
    for(int i = 0; i < Out.rows(); ++i) {
      Out.col(j)[i] = expapprox(Out.col(j)[i]);
      sum[j] += Out.col(j)[i];
    }
    for(int i = 0; i < Out.rows(); ++i) {
      Out.col(j)[i] = logapprox(Out.col(j)[i] / sum[j]);
    }
  }
  return Out;}
}
