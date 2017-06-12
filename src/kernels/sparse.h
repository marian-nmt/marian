#pragma once
#include <cusparse_v2.h>

#include "tensors/tensor.h"

#include "kernels/cuda_helpers.h"

namespace marian {

namespace sparse {

class CSR {
private:
  int nnz_{0};
  int rows_{0};
  int cols_{0};
  size_t device_{0};

  cusparseHandle_t handle_{0};
  cusparseMatDescr_t descr_{0};

  int* rowIndices_{0};
  int* colIndices_{0};
  float* values_{0};

public:
  CSR(int rows, int cols, size_t device)
      : rows_(rows), cols_(cols), device_(device) {
    cudaSetDevice(device);
    CUSPARSE_CHECK(cusparseCreate(&handle_));
    CUSPARSE_CHECK(cusparseCreateMatDescr(&descr_));
    CUSPARSE_CHECK(cusparseSetMatType(descr_, CUSPARSE_MATRIX_TYPE_GENERAL));
    CUSPARSE_CHECK(cusparseSetMatIndexBase(descr_, CUSPARSE_INDEX_BASE_ZERO));
  }

  CSR(int rows,
      int cols,
      const std::vector<float>& values,
      const std::vector<int>& rowIndices,
      const std::vector<int>& colIndices,
      size_t device)
      : nnz_(values.size()), rows_(rows), cols_(cols), device_(device) {
    cudaSetDevice(device);
    CUSPARSE_CHECK(cusparseCreate(&handle_));
    CUSPARSE_CHECK(cusparseCreateMatDescr(&descr_));
    CUSPARSE_CHECK(cusparseSetMatType(descr_, CUSPARSE_MATRIX_TYPE_GENERAL));
    CUSPARSE_CHECK(cusparseSetMatIndexBase(descr_, CUSPARSE_INDEX_BASE_ZERO));

    allocValues();
    allocRowIndices();
    allocColIndices();

    int* cooRowIndices;
    CUDA_CHECK(cudaMalloc(&cooRowIndices, sizeof(int) * nnz_));

    CUDA_CHECK(cudaMemcpy(
        values_, values.data(), sizeof(float) * nnz_, cudaMemcpyDefault));
    CUDA_CHECK(cudaMemcpy(cooRowIndices,
                          rowIndices.data(),
                          sizeof(int) * nnz_,
                          cudaMemcpyDefault));
    CUDA_CHECK(cudaMemcpy(
        colIndices_, colIndices.data(), sizeof(int) * nnz_, cudaMemcpyDefault));

    // @TODO: sort here!

    CUSPARSE_CHECK(cusparseXcoo2csr(handle_,
                                    cooRowIndices,
                                    nnz_,
                                    rows_,
                                    rowIndices_,
                                    CUSPARSE_INDEX_BASE_ZERO));
    CUDA_CHECK(cudaFree(cooRowIndices));
  }

  CSR(Tensor dense) : device_(dense->getDevice()) {
    cudaSetDevice(dense->getDevice());
    rows_ = dense->shape()[0] * dense->shape()[2] * dense->shape()[3];
    cols_ = dense->shape()[1];

    cusparseCreate(&handle_);
    cusparseCreateMatDescr(&descr_);
    cusparseSetMatType(descr_, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr_, CUSPARSE_INDEX_BASE_ZERO);

    // invert everything because of row-major format and use dense2csc,
    // next keep using routines for csr
    int* nnzPerCol;
    cudaMalloc(&nnzPerCol, sizeof(int) * cols_);
    cusparseSnnz(handle_,
                 CUSPARSE_DIRECTION_COLUMN,
                 cols_,
                 rows_,
                 descr_,
                 dense->data(),
                 cols_,
                 nnzPerCol,
                 &nnz_);
    allocValues();
    allocRowIndices();
    allocColIndices();

    cusparseSdense2csc(handle_,
                       cols_,
                       rows_,
                       descr_,
                       dense->data(),
                       cols_,
                       nnzPerCol,
                       values_,
                       colIndices_,
                       rowIndices_);
    cudaFree(nnzPerCol);
  }

  ~CSR() {
    cudaSetDevice(device_);
    if(values_)
      CUDA_CHECK(cudaFree(values_));
    if(rowIndices_)
      CUDA_CHECK(cudaFree(rowIndices_));
    if(colIndices_)
      CUDA_CHECK(cudaFree(colIndices_));

    if(handle_)
      CUSPARSE_CHECK(cusparseDestroy(handle_));
    if(descr_)
      CUSPARSE_CHECK(cusparseDestroyMatDescr(descr_));
  }

  void toTensor(Tensor dense) {
    cudaSetDevice(device_);
    UTIL_THROW_IF2(dense->size() != rows_ * cols_, "Matrix sizes do not match");

    cusparseScsc2dense(handle_,
                       cols_,
                       rows_,
                       descr_,
                       values_,
                       colIndices_,
                       rowIndices_,
                       dense->data(),
                       cols_);
  }

  cusparseHandle_t handle() { return handle_; };
  cusparseMatDescr_t description() { return descr_; };

  int nnz() { return nnz_; }
  int rows() { return rows_; }
  int cols() { return cols_; }

  float* values() { return values_; }
  int* rowIndices() { return rowIndices_; }
  int* colIndices() { return colIndices_; }

  size_t getDevice() { return device_; }

  void allocValues(int nnz = 0) {
    cudaSetDevice(device_);
    if(nnz > 0)
      nnz_ = nnz;
    if(values_)
      CUDA_CHECK(cudaFree(values_));
    CUDA_CHECK(cudaMalloc(&values_, sizeof(float) * nnz_));
  }

  void allocRowIndices(int rows = 0) {
    cudaSetDevice(device_);
    if(rows > 0)
      rows_ = rows;
    if(rowIndices_)
      CUDA_CHECK(cudaFree(rowIndices_));
    CUDA_CHECK(cudaMalloc(&rowIndices_, sizeof(int) * (rows_ + 1)));
  }

  void allocColIndices(int nnz = 0) {
    cudaSetDevice(device_);
    if(nnz > 0)
      nnz_ = nnz;
    if(colIndices_)
      CUDA_CHECK(cudaFree(colIndices_));
    CUDA_CHECK(cudaMalloc(&colIndices_, sizeof(int) * nnz_));
  }

  std::string debug() {
    uint8_t* buffer;
    cudaMalloc(&buffer, sizeof(float) * rows() * cols());

    auto mem = New<MemoryPiece>(buffer, sizeof(float) * rows() * cols());
    Tensor tensor(new TensorBase(mem, {rows(), cols()}, device_));
    toTensor(tensor);
    std::string temp = tensor->debug();

    cudaFree(buffer);
    return temp;
  }
};

void multiply(
    Ptr<CSR>, const Ptr<CSR>, const Ptr<CSR>, bool = false, bool = false);

void LfaForward(Tensor out, Tensor logits, Tensor att, Ptr<CSR> sparseLf);

void LfaBackward(Tensor grad, Tensor adj, Ptr<CSR> sparseLf);
}
}
