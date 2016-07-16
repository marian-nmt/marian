#include "matrix.h"
#include "simd_math_prims.h"

namespace mblas {

Matrix& Swap(Matrix& Out, Matrix& In) {
  size_t iRows = In.Rows();
  size_t iCols = In.Cols();
  size_t oRows = Out.Rows();
  size_t oCols = Out.Cols();

  Out.Reshape(iRows, iCols);
  In.Reshape(oRows, oCols);

  In.GetVec().swap(Out.GetVec());
  return Out;
}

Matrix& Mean(Matrix& Out, const Matrix& In) {
  size_t m = In.Rows();
  size_t n = In.Cols();

  Out.Resize(1, n, 0.f);
  Matrix Ones(1, m, 1.f);

  float alpha = 1.0 / m;
  float beta  = 0.0;
  cblas_sgemv(CblasColMajor, CblasNoTrans, n, m, alpha, In.data(), n,
              Ones.data(), 1, beta, Out.data(), 1);
  return Out;
}

Matrix& Transpose(Matrix& Out, const Matrix& In) {
  size_t m = In.Rows();
  size_t n = In.Cols();

  Out.Resize(n, m);

  const float* d_in = In.data();
  float* d_out = Out.data();
  
  for(int i = 0; i < m; ++i)
    for(int j = 0; j < n; ++j)
      d_out[j * m + i] = d_in[i * n  + j];
    
  return Out;
}

Matrix& Transpose(Matrix& Out) {
  Matrix Temp;
  Transpose(Temp, Out);
  Swap(Out, Temp);
  return Out;
}

Matrix& Copy(Matrix& Out, const Matrix& In) {
  Out.Resize(In.Rows(), In.Cols());
  std::copy(In.begin(), In.end(), Out.begin());
  return Out;
}

Matrix& PasteRow(Matrix& Out,
                 const Matrix& In,
                 const size_t r, const size_t c) {
  size_t start = r * Out.Cols() + c;
  std::copy(In.begin(), In.end(), Out.begin() + start);
  return Out;
}

Matrix& CopyRow(Matrix& Out,
                const Matrix& In,
                const size_t r, const size_t c) {
  size_t length = In.Cols() - c;
  Out.Resize(1, length);
  size_t start = r * In.Cols() + c;
  size_t end   = start + length;
  std::copy(In.begin() + start, In.begin() + end, Out.begin());
  return Out;
}

void gCopyRows(float* out, const float* in, size_t cols,
               const RowPair* devPairs, size_t numPairs) {
  for(int j = 0; j < numPairs; ++j) {
      size_t dstId = devPairs[j].first;
      size_t srcId = devPairs[j].second;

      float* rowOut = out + dstId * cols;
      const float* rowIn = in + srcId * cols;

      for(int i = 0; i < cols; ++i)
          rowOut[i] = rowIn[i];
  }
}

Matrix& CopyRows(Matrix& Out,
                 const Matrix& In,
                 const RowPair* devPairs,
                 size_t numPairs) {
  float* d_out = Out.data();
  const float* d_in = In.data();

  gCopyRows(d_out, d_in, In.Cols(), devPairs, numPairs);
  return Out;
}

Matrix& CopyRows(Matrix& Out,
                 const Matrix& In,
                 const RowPairs& pairs) {
  CopyRows(Out, In, pairs.data(), pairs.size());
  return Out;
}

Matrix& Concat(Matrix& Out, const Matrix& In) {
  size_t oldSize = Out.size();
  Out.Resize(Out.Rows() + In.Rows(), Out.Cols());
  std::copy(In.begin(), In.end(), Out.begin() + oldSize);
  return Out;
}

Matrix& Assemble(Matrix& Out,
                 const Matrix& In,
                 const std::vector<size_t>& indeces) {
  RowPairs rowPairs;
  for(size_t i = 0; i < indeces.size(); i++)
    rowPairs.emplace_back(i, indeces[i]);
  Out.Resize(rowPairs.size(), In.Cols());
  CopyRows(Out, In, rowPairs);
  return Out;
}

Matrix& AssembleCols(Matrix& Out,
                 const Matrix& In,
                 const std::vector<size_t>& indeces) {
  Out.Resize(In.Rows(), indeces.size());
  
  size_t colsSrc = In.Cols();
  size_t colsDst = Out.Cols();
  for(size_t i = 0; i < Out.Rows(); ++i) {
    for(size_t j = 0; j < indeces.size(); ++j) {
      size_t src = indeces[j];
      size_t dst = j;  
      Out.data()[i * colsDst + dst] = In.data()[i * colsSrc + src]; 
    }
  }
  return Out;
}

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

Matrix& Slice(Matrix& Out,
              const Matrix& In,
              size_t n, size_t dim) {

  Out.Resize(In.Rows(), dim);

  float* d_out = Out.data();
  const float* d_in = In.data();

  gSlice(d_out, d_in, n, dim, In.Rows(), In.Cols());
  return Out;
}

Matrix& Prod(Matrix& C, const Matrix& A, const Matrix& B,
             bool transA, bool transB) {
  Matrix::value_type alpha = 1.0;
  Matrix::value_type beta = 0.0;
  
  size_t m = A.Rows();
  size_t k = A.Cols();
  if(transA)
    std::swap(m, k);
  
  size_t l = B.Rows();
  size_t n = B.Cols();
  if(transB)
    std::swap(l, n);
  
  size_t lda = A.Cols();
  size_t ldb = B.Cols();
  size_t ldc = B.Cols();
  
  if(transB)
    ldc = B.Rows();
  
  C.Resize(m, n);
  
  auto opA = transA ? CblasTrans : CblasNoTrans;
  auto opB = transB ? CblasTrans : CblasNoTrans;
  
  cblas_sgemm(CblasColMajor, opB, opA,
              n, m, k, alpha, B.data(), ldb, A.data(), lda, beta, C.data(), ldc);
  return C;
}

void gSoftMax(float* d, size_t rows, size_t cols) {
  float sum[rows];
  for(int j = 0; j < rows; ++j) {
    sum[j] = 0;
    float* out = d + j * cols;
    for(int i = 0; i < cols; ++i) {
      out[i] = expapprox(out[i]);
      sum[j]+= out[i];
    }
    for(int i = 0; i < cols; ++i) {
      out[i] /= sum[j];
    }
  }
}

Matrix& Softmax(Matrix& Out) {
  gSoftMax(Out.data(), Out.Rows(), Out.Cols());
  return Out;
}

void gSoftMaxLog(float* d, size_t rows, size_t cols) {
  float sum[rows];
  for(int j = 0; j < rows; ++j) {
    sum[j] = 0;
    float* out = d + j * cols;
    for(int i = 0; i < cols; ++i) {
      out[i] = expapprox(out[i]);
      sum[j]+= out[i];
    }
    for(int i = 0; i < cols; ++i) {
      out[i] = logapprox(out[i] / sum[j]);
    }
  }
}

Matrix& SoftmaxLog(Matrix& Out) {
  gSoftMaxLog(Out.data(), Out.Rows(), Out.Cols());
}


}
