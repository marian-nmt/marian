#include "kernels/sparse.h"
#include "kernels/tensor_operators.h"
#include "kernels/thrust_functions.h"
#include "tensors/tensor.h"

namespace marian {

namespace sparse {

void multiply(
    Ptr<CSR> C, const Ptr<CSR> A, const Ptr<CSR> B, bool transA, bool transB) {
  cudaSetDevice(C->getDevice());
  int nnzTotal;
  C->allocRowIndices(A->rows());
  CUSPARSE_CHECK(cusparseXcsrgemmNnz(
      A->handle(),
      transA ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE,
      transB ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE,
      A->rows(),
      B->cols(),
      A->cols(),
      A->description(),
      A->nnz(),
      A->rowIndices(),
      A->colIndices(),
      B->description(),
      B->nnz(),
      B->rowIndices(),
      B->colIndices(),
      C->description(),
      C->rowIndices(),
      &nnzTotal));

  C->allocValues(nnzTotal);
  C->allocColIndices(nnzTotal);
  CUSPARSE_CHECK(cusparseScsrgemm(
      A->handle(),
      transA ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE,
      transB ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE,
      A->rows(),
      B->cols(),
      A->cols(),
      A->description(),
      A->nnz(),
      A->values(),
      A->rowIndices(),
      A->colIndices(),
      B->description(),
      B->nnz(),
      B->values(),
      B->rowIndices(),
      B->colIndices(),
      C->description(),
      C->values(),
      C->rowIndices(),
      C->colIndices()));
}

//__global__ void gExpandAtt(float* out,
//                           const float* in,
//                           int batch,
//                           int srcWords,
//                           int nonzeros) {
//
//  for(int bid = 0; bid < nonzeros; bid += blockDim.x * gridDim.x) {
//    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
//    if (index < nonzeros) {
//      int r = (index % batch) + (index / (srcWords * batch)) * batch;
//      int c = index % (srcWords * batch);
//      out[r * srcWords * batch + c] = in[index];
//    }
//  }
//}
//
//
// void ExpandAtt(Tensor out, Tensor in) {
//  cudaSetDevice(in->getDevice());
//  int nonzeros = in->shape().elements();
//  int batch = in->shape()[0];
//  int srcWords = in->shape()[2];
//
//  int threads = std::min(MAX_THREADS, nonzeros);
//  int blocks  = std::min(MAX_BLOCKS, nonzeros / threads  + (nonzeros % threads
//  != 0));
//
//  gCollapseAtt<<<blocks, threads>>>(out->data(), in->data(), batch, srcWords,
//  nonzeros);
//}

void LfaForward(Tensor out, Tensor logits, Tensor att, Ptr<CSR> sparseLf) {
  cudaSetDevice(out->getDevice());

  int batch = att->shape()[0];
  int srcWords = att->shape()[2];
  int trgWords = att->shape()[3];

  std::vector<float> values;
  att->get(values);
  int nonzeros = values.size();
  std::vector<std::tuple<int, int, float>> coo;
  for(size_t i = 0; i < nonzeros; ++i) {
    int r = (i % batch) + (i / (srcWords * batch)) * batch;
    int c = i % (srcWords * batch);
    UTIL_THROW_IF2(r >= trgWords * batch, "Row index too large");
    UTIL_THROW_IF2(c >= srcWords * batch, "Column index too large");
    coo.emplace_back(r, c, values[i]);
  }
  std::sort(coo.begin(), coo.end());
  values.clear();
  values.resize(nonzeros);
  std::vector<int> rowInd(nonzeros);
  std::vector<int> colInd(nonzeros);
  for(int i = 0; i < nonzeros; ++i) {
    rowInd[i] = std::get<0>(coo[i]);
    colInd[i] = std::get<1>(coo[i]);
    values[i] = std::get<2>(coo[i]);
  }

  auto sparseAtt = New<CSR>(batch * trgWords,
                            batch * srcWords,
                            values,
                            rowInd,
                            colInd,
                            out->getDevice());

  auto sparseLfa
      = New<CSR>(sparseAtt->rows(), sparseLf->cols(), out->getDevice());
  multiply(sparseLfa, sparseAtt, sparseLf);

  sparseLfa->toTensor(out);
}

__global__ void gCollapseAtt(
    float* out, const float* in, int batch, int srcWords, int nonzeros) {
  for(int bid = 0; bid < nonzeros; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if(index < nonzeros) {
      int r = (index % batch) + (index / (srcWords * batch)) * batch;
      int c = index % (srcWords * batch);
      float val = in[r * srcWords * batch + c];
      out[index] += val;
    }
  }
}

void CollapseAtt(Tensor out, Tensor in) {
  cudaSetDevice(out->getDevice());
  int nonzeros = out->shape().elements();
  int batch = out->shape()[0];
  int srcWords = out->shape()[2];

  int threads = std::min(MAX_THREADS, nonzeros);
  int blocks
      = std::min(MAX_BLOCKS, nonzeros / threads + (nonzeros % threads != 0));

  gCollapseAtt<<<blocks, threads>>>(
      out->data(), in->data(), batch, srcWords, nonzeros);
}

void LfaBackward(Tensor gradAtt, Tensor adj, Ptr<CSR> sparseLf) {
  cudaSetDevice(adj->getDevice());

  int batch = gradAtt->shape()[0];
  int srcWords = gradAtt->shape()[2];
  int trgWords = gradAtt->shape()[3];
  int nonzeros = gradAtt->shape().elements();

  int dimTrgVoc = adj->shape()[1];

  int exSize = sizeof(float) * batch * srcWords * batch * trgWords;
  uint8_t* expandAttGradBuffer;
  CUDA_CHECK(cudaMalloc(&expandAttGradBuffer,
                        exSize));

  float alpha = 1, beta = 0;
  CUSPARSE_CHECK(cusparseScsrmm2(sparseLf->handle(),
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 sparseLf->rows(),
                                 batch * trgWords,
                                 sparseLf->cols(),
                                 sparseLf->nnz(),
                                 &alpha,
                                 sparseLf->description(),
                                 sparseLf->values(),
                                 sparseLf->rowIndices(),
                                 sparseLf->colIndices(),
                                 adj->data(),
                                 dimTrgVoc,
                                 &beta,
                                 (float*)expandAttGradBuffer,
                                 batch * srcWords));

  Tensor expandAttGrad(new TensorBase(
      New<MemoryPiece>(expandAttGradBuffer, exSize), {batch * trgWords, batch * srcWords}, 0));
  CollapseAtt(gradAtt, expandAttGrad);
  CUDA_CHECK(cudaFree(expandAttGradBuffer));
}
}
}
