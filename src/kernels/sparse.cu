#include "kernels/sparse.h"
#include "kernels/tensor_operators.h"
#include "kernels/thrust_functions.h"
#include "tensors/tensor.h"

namespace marian {
  
namespace sparse {
  
void multiply(Ptr<CSR> C, const Ptr<CSR> A, const Ptr<CSR> B,
              bool transA, bool transB) {
  cudaSetDevice(C->getDevice());
  int nnzTotal;
  C->allocRowIndices(A->rows());
  CUSPARSE_CHECK(cusparseXcsrgemmNnz(A->handle(),
                      transA ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE,
                      transB ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE,
                      A->rows(), B->cols(), A->cols(),
                      A->description(), A->nnz(), A->rowIndices(), A->colIndices(),
                      B->description(), B->nnz(), B->rowIndices(), B->colIndices(),
                      C->description(), C->rowIndices(), &nnzTotal));
  
  C->allocValues(nnzTotal);
  C->allocColIndices(nnzTotal); 
  CUSPARSE_CHECK(cusparseScsrgemm(A->handle(),
                   transA ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE,
                   transB ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE,
                   A->rows(), B->cols(), A->cols(),
                   A->description(), A->nnz(), A->values(), A->rowIndices(), A->colIndices(),
                   B->description(), B->nnz(), B->values(), B->rowIndices(), B->colIndices(),
                   C->description(), C->values(), C->rowIndices(), C->colIndices()));
}

void LfaForward(Tensor out, Tensor logits, Tensor att,
                Ptr<CSR> sparseLf, float eps) {
  cudaSetDevice(out->getDevice());
  
  size_t batch    = att->shape()[0];
  size_t srcWords = att->shape()[2];
  size_t trgWords = att->shape()[3];
  
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
  
  auto sparseAtt = New<CSR>(batch * trgWords, batch * srcWords,
                            values, rowInd, colInd, out->getDevice());
  
  auto sparseLfa = New<CSR>(sparseAtt->rows(), sparseLf->cols(), out->getDevice());
  multiply(sparseLfa, sparseAtt, sparseLf);
  
  sparseLfa->toTensor(out);
  Element(_1 = (Log(_1 + eps) + _2), out, logits);
}

void LfaBackward(Tensor gradAtt,
                 Tensor valLogits, Tensor valAtt,
                 Tensor adj,
                 Ptr<CSR> sparseLf, float eps) {
  cudaSetDevice(adj->getDevice());
  
  //int batch    = gradAtt->shape()[0];
  //int srcWords = gradAtt->shape()[2];
  //int trgWords = gradAtt->shape()[3];
  //int nonzeros = gradAtt->shape().elements();
  //
  //int dimTrgVoc = valLogits->shape()[1];
  //
  //Element(_1 = _2 / (_1 + eps), valLogits, adj);
  //
  //std::cerr << valLogits->debug() << std::endl;
  //
  //float* buffer;
  //CUDA_CHECK(cudaMalloc(&buffer, sizeof(float) * batch * srcWords * batch * trgWords));
  //
  //float alpha = 1, beta = 0;
  //CUSPARSE_CHECK(cusparseScsrmm2(sparseLf->handle(),
  //  CUSPARSE_OPERATION_NON_TRANSPOSE,
  //  CUSPARSE_OPERATION_NON_TRANSPOSE,
  //  sparseLf->rows(), batch * trgWords, sparseLf->cols(), sparseLf->nnz(), &alpha,
  //  sparseLf->description(), sparseLf->values(), sparseLf->rowIndices(), sparseLf->colIndices(),
  //  adj->data(), dimTrgVoc, &beta, buffer, batch * srcWords));
  //
  //Tensor b(new TensorBase(buffer, {batch * trgWords, batch * srcWords}, 0));
  //std::cerr << b->debug() << std::endl;
  //
  //CUDA_CHECK(cudaFree(buffer));
}

}

}
