#include "gpu/mblas/matrix_functions.h"
#include "gpu/mblas/handles.h"

using namespace std;

namespace amunmt {
namespace GPU {
namespace mblas {

thread_local cublasHandle_t* CublasHandler::handle_ = nullptr;
thread_local CudaStreamHandler* CudaStreamHandler::instance_ = nullptr;;

Matrix& Swap(Matrix& Out, Matrix& In) {
  Out.swap(In);
  return Out;
}

__global__ void gMean(MatrixWrapper<float> out,
                      const MatrixWrapper<float> in,
                      const MatrixWrapper<int>  mapping)
{
  assert(out.dim(0) == 1);
  // in = max sentence length, whatever, 1, batches
  // out = in, dim(0 = 1
  // mapping = max length * batches

  int id = threadIdx.x + blockIdx.x * blockDim.x;
  //printf("id = %d in = %lu %lu %lu %lu = %lu %lu \n", id, in.dim(0), in.dim(1), in.dim(2), in.dim(3), in.size(), sizeof(in));

  if (id < out.size()) {
    size_t indices[SHAPE_SIZE];
    out.id2Indices(id, indices);
    assert(indices[0] == 0);
    //printf("%d -> %lu %lu %lu %lu \n", id, indices[0], indices[1], indices[2], indices[3]);

    size_t batch = indices[3];
    size_t startMapInd = batch * in.dim(0);

    float sum = 0.0f;
    int counter = 0;
    for (size_t row = 0; row < in.dim(0); ++row) {
      int isWord = mapping(row, batch, 0, 0);
      //printf("batch=%lu startMapInd=%lu  mapOffset=%lu -> %d \n", batch, startMapInd, mapOffset, isWord);
      if (isWord) {
        sum += in(row, indices[1], indices[2], indices[3]);
        ++counter;
      }
    }

    sum /= (float) counter;
    out[id] = sum;
  }
}

void Mean(Matrix& Out, const Matrix& In, const DeviceVector<int>& mapping) {
  size_t batchNum = Out.dim(0) * Out.dim(2) * Out.dim(3);
  size_t stateLength = Out.dim(1);
  size_t sentenceLength = (In.dim(0) * In.dim(2) * In.dim(3)) / batchNum;

  MatrixWrapper<float> outWrap(Out);
  MatrixWrapper<float> inWrap(In);

  MatrixWrapper<int> mappingWrap(mapping, sentenceLength, batchNum, 1, 1);

  size_t threads = MAX_THREADS;
  size_t blocks =  (outWrap.size() / threads) + ((outWrap.size() % threads == 0) ?  0 : 1);

  gMean<<<blocks, threads, 0, CudaStreamHandler::GetStream()>>>
    (outWrap, inWrap, mappingWrap);

}

__global__ void gWeightedMeanOld(float* d_out, const float* weights, const float* d_in, const int* mapping,
                              int numRows, int numCols, int srcLen) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id < numRows * numCols) {
    int rowNo = id / numCols;
    int batchNo = mapping[rowNo];
    int statePos = id % numCols;

    float sum = 0.0f;
    for (int i = 0; i < srcLen; ++i) {
      sum += weights[rowNo * srcLen + i] * d_in[batchNo * srcLen * numCols + (i * numCols) + statePos];
    }

    d_out[id] = sum;
  }
}


__global__ void gWeightedMean(MatrixWrapper<float> out,
                              const MatrixWrapper<float> weights,
                              const MatrixWrapper<float> in,
                              const MatrixWrapper<int> mapping
                              )
{
  int numHypos = weights.dim(0);
  int states = in.dim(1);
  int srcLen = weights.dim(1);

  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id < numHypos * states) {
    int hypoInd = id / states;
    int batchInd = mapping[hypoInd];
    int stateInd = id % states;
    //printf("hypoInd=%d batchInd=%d stateInd=%d \n", hypoInd, batchInd, stateInd);

    float sum = 0.0f;
    for (uint i = 0; i < srcLen; ++i) {
      sum += weights(hypoInd, i, 0, 0) * in(i, stateInd, 0, batchInd);
    }

    out[id] = sum;
  }
}

void WeightedMean(Matrix& Out,const Matrix& Weights, const Matrix& In, const DeviceVector<int>& mapping) {
  int numHypos = Weights.dim(0);
  int states = In.dim(1);

  Out.Resize(numHypos, states);

  MatrixWrapper<float> outWrap(Out);
  MatrixWrapper<float> weightsWrap(Weights);
  MatrixWrapper<float> inWrap(In);
  MatrixWrapper<int> mappingWrap(mapping);

  int nThreads = MAX_THREADS;
  int nBlocks =  (Out.size() / MAX_THREADS) + ((Out.size() % MAX_THREADS == 0) ?  0 : 1);

  gWeightedMean<<<nBlocks, nThreads, 0, CudaStreamHandler::GetStream()>>>
    (outWrap, weightsWrap, inWrap, mappingWrap);
  /*
  cerr << "nBlocks=" << nBlocks << endl;

  cerr << "Out=" << outWrap.Debug() << endl;
  cerr << "Weights=" << weightsWrap.Debug() << endl;
  cerr << "In=" << inWrap.Debug() << endl;
  cerr << "mapping=" << mapping.size() << endl;
  for (size_t i = 0; i < mapping.size(); ++i) {
    cerr << mapping[i] << " ";
  }
  cerr << endl << endl;
  */
}

Matrix& Transpose(Matrix& Out, const Matrix& In) {
  size_t m = In.dim(0);
  size_t n = In.dim(1);

  Out.Resize(n, m);

  float alpha = 1.0;
  float beta  = 0.0;

  cublasSgeam(CublasHandler::GetHandle(), CUBLAS_OP_T, CUBLAS_OP_T, m, n, &alpha, In.data(), n,
              &beta, In.data(), n, Out.data(), m);

  return Out;
}

Matrix& Transpose(Matrix& Out) {
  Matrix Temp;
  Transpose(Temp, Out);
  Swap(Out, Temp);
  return Out;
}

Matrix& Concat(Matrix& Out, const Matrix& In) {
  size_t oldSize = Out.size();
  Out.Resize(Out.dim(0) + In.dim(0), Out.dim(1));

  mblas::copy(In.data(), In.size(), Out.data() + oldSize, cudaMemcpyDeviceToDevice);

  return Out;
}

Matrix& Copy(Matrix& Out, const Matrix& In) {
  Out.Resize(In.dim(0), In.dim(1), In.dim(2), In.dim(3));

  mblas::copy(In.data(), In.size(), Out.data(), cudaMemcpyDeviceToDevice);

  return Out;
}

__global__ void gPasteRows(  MatrixWrapper<float> outWrap,
                          const MatrixWrapper<float> inWrap,
                          int rowNo, int colNo)
{
  int inRows = inWrap.dim(0);
  int inCols = inWrap.dim(1);

  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id < inRows * inCols) {
    int outCols = outWrap.dim(1);

    int inRow = id / inCols;
    int inCol = id % inCols;

    //outWrap[outID] = inWrap[id];
    outWrap(rowNo, inCol + colNo, 0, inRow) = inWrap(inRow, inCol, 0, 0);
  }
}

void PasteRows(Matrix& Out, const Matrix& In, const size_t rowNo, size_t colNo)
{
  MatrixWrapper<float> outWrap(Out);
  MatrixWrapper<float> inWrap(In);

  int nThreads = MAX_THREADS;
  int nBlocks =  (In.size() / 512) + ((In.size() % 512 == 0) ?  0 : 1);

  gPasteRows<<<nBlocks, nThreads, 0, CudaStreamHandler::GetStream()>>>
    (outWrap, inWrap, rowNo, colNo);

}

Matrix& PasteRow(Matrix& Out,
                 const Matrix& In,
                 const size_t r, const size_t c)
{
  size_t start = r * Out.dim(1) + c;

  mblas::copy(In.data(), In.size(), Out.data() + start, cudaMemcpyDeviceToDevice);

  return Out;
}

Matrix& CopyRow(Matrix& Out,
                const Matrix& In,
                const size_t r, const size_t c) {
  size_t length = In.dim(1) - c;
  Out.Resize(1, length);
  size_t start = r * In.dim(1) + c;
  //size_t end   = start + length;

  //mblas::copy(In.begin() + start, In.begin() + end, Out.begin());
  mblas::copy(In.data() + start, length , Out.data(), cudaMemcpyDeviceToDevice);

  return Out;
}

__global__ void gCopyRows(MatrixWrapper<float> outWrap,
                          const MatrixWrapper<float> inWrap,
                          const MatrixWrapper<size_t> indicesWrap,
                          float* out, const float* in)
{
  size_t numPairs = indicesWrap.size();
  size_t cols = inWrap.dim(1);

  size_t indicesInd = blockIdx.x;
  size_t inRow =indicesWrap[indicesInd];

  size_t colInd = threadIdx.x;
  while (colInd < outWrap.dim(1)) {
	  outWrap(indicesInd, colInd, 0, 0) = inWrap(inRow, colInd, 0, 0);
	  colInd += gridDim.x;
  }
}

Matrix& CopyRows(Matrix& Out,
                 const Matrix& In,
                 const DeviceVector<size_t>& indices)
{
  float* d_out = Out.data();
  const float* d_in = In.data();

  size_t numPairs = indices.size();

  MatrixWrapper<float> outWrap(Out);
  const MatrixWrapper<float> inWrap(In);
  const MatrixWrapper<size_t> indicesWrap(indices);

  int threads = std::min(MAX_THREADS, (int)In.dim(1));
  int blocks = std::min(MAX_BLOCKS, (int)numPairs);

  gCopyRows<<<blocks, threads, 0, CudaStreamHandler::GetStream()>>>
    (outWrap, inWrap, indicesWrap, d_out, d_in);

  return Out;
}


Matrix& Assemble(Matrix& Out,
                 const Matrix& In,
                 const DeviceVector<size_t>& indices) {
  Out.Resize(indices.size(), In.dim(1));
  //cerr << "Assemble=" << Out.Debug() << " " << In.Debug() << indices.size() << endl;

  CopyRows(Out, In, indices);
  return Out;
}

__global__ void gSlice(MatrixWrapper<float> outWrap,
						          const MatrixWrapper<float> inWrap,
                       size_t n, size_t dim)
{
  size_t row = blockIdx.x;

  size_t inCol = threadIdx.x + dim * n;
  size_t outCol = threadIdx.x;

  while (outCol < outWrap.dim(1)) {
    outWrap(row, outCol, 0, 0) = inWrap(row, inCol, 0, 0);

    inCol += gridDim.x;
    outCol += gridDim.x;
  }

}

Matrix& Slice(Matrix& Out,
              const Matrix& In,
              size_t n, size_t dim) {

  Out.Resize(In.dim(0), dim);

  MatrixWrapper<float> outWrap(Out);
  const MatrixWrapper<float> inWrap(In);

  int threads = std::min(MAX_THREADS, (int)dim);
  int blocks = std::min(MAX_BLOCKS, (int)In.dim(0));

  gSlice<<<blocks, threads, 0, CudaStreamHandler::GetStream()>>>
    (outWrap, inWrap, n, dim);

  return Out;
}

Matrix& Prod(cublasHandle_t handle, Matrix& C, const Matrix& A, const Matrix& B,
             bool transA, bool transB) {
  Matrix::value_type alpha = 1.0;
  Matrix::value_type beta = 0.0;

  size_t m = A.dim(0);
  size_t k = A.dim(1);
  if(transA)
    std::swap(m, k);

  size_t l = B.dim(0);
  size_t n = B.dim(1);
  if(transB)
    std::swap(l, n);

  size_t lda = A.dim(1);
  size_t ldb = B.dim(1);
  size_t ldc = B.dim(1);

  if(transB)
    ldc = B.dim(0);

  C.Resize(m, n, A.dim(2), A.dim(3));
  //cerr << "C=" << C.Debug(1) << endl;

  cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

  size_t m2 = A.dim(0) * A.dim(2) * A.dim(3);

  cublasSgemm(handle, opB, opA,
              n, m2, k, &alpha, B.data(), ldb, A.data(), lda, &beta, C.data(), ldc);
  return C;
}

Matrix& Prod(Matrix& C, const Matrix& A, const Matrix& B,
             bool transA, bool transB) {

  //std::cerr << "1C=" << C.Debug() << std::endl;
  //std::cerr << "1A=" << A.Debug() << std::endl;
  //std::cerr << "1B=" << B.Debug() << std::endl;

  Matrix &ret = Prod(CublasHandler::GetHandle(), C, A, B, transA, transB);

  //std::cerr << "2C=" << C.Debug() << std::endl;
  return ret;
}

__global__ void gSoftMax(MatrixWrapper<float> outWrap,
                         const MatrixWrapper<int> batchIdsWrap,
                         const MatrixWrapper<int> srcMappingWrap)
{
  extern __shared__ float _share[];

  size_t numHypos = outWrap.dim(0);
  size_t srcLen = outWrap.dim(1);

  int hypoInd =  blockIdx.x;
  int origSrcPos = threadIdx.x;

  while (hypoInd < numHypos) {
    float* _max = _share;
    _max[origSrcPos] = outWrap(hypoInd, origSrcPos, 0, 0);
    for (int tid = 0; tid < srcLen; tid += blockDim.x) {
      int srcPos = tid + origSrcPos;
      if (srcPos < srcLen) {
        float value = outWrap(hypoInd, srcPos, 0, 0);

        int batch = batchIdsWrap[hypoInd];
        value *= srcMappingWrap(srcPos, batch, 0, 0);
        if (value > _max[origSrcPos]) {
          _max[origSrcPos] = value;
        }
      }
    }

    int len = blockDim.x;
    while (len != 1) {
      __syncthreads();

      int skip = (len + 1) >> 1;
      if (origSrcPos < (len >> 1)) {
        if(_max[origSrcPos + skip] > _max[origSrcPos])
          _max[origSrcPos] = _max[origSrcPos + skip];
      }
      len = (len + 1) >> 1;
    }
    __syncthreads();
    float max = _max[0];
    __syncthreads();

    float* _sum = _share;// + blockDim.x;
    _sum[origSrcPos] = 0.0f;
    for (int tid = 0; tid < srcLen; tid += blockDim.x) {
      int srcPos = tid + origSrcPos;
      if (srcPos < srcLen) {
        outWrap(hypoInd, srcPos, 0, 0) = __expf(outWrap(hypoInd, srcPos, 0, 0) - max);

        int batch = batchIdsWrap[hypoInd];
        outWrap(hypoInd, srcPos, 0, 0) *= srcMappingWrap(srcPos, batch, 0, 0);
        _sum[origSrcPos] += outWrap(hypoInd, srcPos, 0, 0);
      }
    }

    __syncthreads();

    len = blockDim.x;
    while (len != 1) {
      __syncthreads();

      int skip = (len + 1) >> 1;
      if (origSrcPos < (len >> 1)) {
        _sum[origSrcPos] += _sum[origSrcPos + skip];
      }
      len = (len + 1) >> 1;
    }

    __syncthreads();

    for (int tid = 0; tid < srcLen; tid += blockDim.x) {
      int srcPos = tid + origSrcPos;
      if (srcPos < srcLen) {
        outWrap(hypoInd, srcPos, 0, 0) /= _sum[0];
      }
    }
    __syncthreads();
    hypoInd += gridDim.x;
  }
}

Matrix& Softmax(Matrix& Out, const DeviceVector<int>& batchIds, const DeviceVector<int>& srcMapping, size_t batchSize)
{
  size_t srcSize = Out.dim(1);

  MatrixWrapper<float> outWrap(Out);
  const MatrixWrapper<int> batchIdsWrap(batchIds);
  const MatrixWrapper<int> srcMappingWrap(srcMapping, srcSize, batchSize, 1, 1);

  int blocks = batchSize;
  int threads = std::min(MAX_THREADS, (int)srcSize);
  int shared = sizeof(float) * threads;

  gSoftMax<<<blocks, threads, shared, CudaStreamHandler::GetStream()>>>
    (outWrap, batchIdsWrap, srcMappingWrap);

  return Out;
}

__global__ void gLogSoftMax(MatrixWrapper<float> outWrap)
{
  extern __shared__ float _share[];

  size_t rows = outWrap.dim(0);
  size_t cols = outWrap.dim(1);

  int rowIdx =  blockIdx.x;

  while (rowIdx < rows) {
    float* _max = _share;
    _max[threadIdx.x] = outWrap(rowIdx, threadIdx.x, 0, 0);
    for (int tid = 0; tid < cols; tid += blockDim.x) {
      int id = tid + threadIdx.x;
      if (id < cols) {
        const float &val = outWrap(rowIdx, id, 0, 0);
        if (val > _max[threadIdx.x]) {
          _max[threadIdx.x] = val;
        }
      }
    }

    int len = blockDim.x;
    while (len != 1) {
      __syncthreads();

      int skip = (len + 1) >> 1;
      if (threadIdx.x < (len >> 1)) {
        if(_max[threadIdx.x + skip] > _max[threadIdx.x])
          _max[threadIdx.x] = _max[threadIdx.x + skip];
      }
      len = (len + 1) >> 1;
    }
    __syncthreads();
    float max = _max[0];
    __syncthreads();

    float* _sum = _share;// + blockDim.x;

    _sum[threadIdx.x] = 0.0f;
    for (int tid = 0; tid < cols; tid += blockDim.x) {
      int id = tid + threadIdx.x;
      if (id < cols) {
        //row[id] = exp(row[id] - max);
        float &val = outWrap(rowIdx, id, 0, 0);
        val = __expf(val - max);
        _sum[threadIdx.x] += val;
      }
    }

    len = blockDim.x;
    while (len != 1) {
      __syncthreads();

      int skip = (len + 1) >> 1;
      if (threadIdx.x < (len >> 1)) {
        _sum[threadIdx.x] += _sum[threadIdx.x + skip];
      }
      len = (len + 1) >> 1;
    }

    __syncthreads();

    for (int tid = 0; tid < cols; tid += blockDim.x) {
      int id = tid + threadIdx.x;
      if (id < cols) {
        //row[id] = log(row[id]/_sum[0]);
        float &val = outWrap(rowIdx, id, 0, 0);
        val = __logf(val /_sum[0]);
      }
    }
    __syncthreads();
    rowIdx += gridDim.x;
  }
}


Matrix& LogSoftmax(Matrix& Out)
{
  MatrixWrapper<float> outWrap(Out);

  int blocks = std::min(MAX_BLOCKS, (int)Out.dim(0));
  int threads = std::min(MAX_THREADS, (int)Out.dim(1));
  int shared = sizeof(float) * threads;

  gLogSoftMax<<<blocks, threads, shared, CudaStreamHandler::GetStream()>>>
    (Out);

  return Out;
}

__global__ void gSetColumn(MatrixWrapper<float> inWrap, int noColumn, float value) {
  int n_rows = inWrap.dim(0);

  int rowNumber = threadIdx.x  + blockDim.x * blockIdx.x;

  if (rowNumber < n_rows) {
    inWrap(rowNumber, noColumn, 0, 0) = value;
  }
}

void SetColumn(Matrix& In, int noColumn, float value) {
  int nRows = In.dim(0);
  int nBlocks = nRows / MAX_THREADS + ((nRows % MAX_THREADS == 0) ?  0 : 1);
  int nThreads = std::min(MAX_THREADS, nRows);

  MatrixWrapper<float> inWrap(In);

  gSetColumn<<<nBlocks, nThreads, 0, mblas::CudaStreamHandler::GetStream()>>>
    (inWrap, noColumn, value);
}

__global__ void gFill(MatrixWrapper<float> inWrap, float val) {
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  if (index < inWrap.size()) {
    inWrap[index] = val;
  }
}

void Fill(Matrix& In, float value) {
  size_t size = In.size();

  if (value) {
    int nThreads = std::min(MAX_THREADS, (int)size);
    int nBlocks = (size / nThreads) + ((size % nThreads == 0) ? 0 : 1);

    MatrixWrapper<float> inWrap(In);

    gFill<<<nBlocks, nThreads, 0, CudaStreamHandler::GetStream()>>>
      (inWrap, value);
  }
  else {
    HANDLE_ERROR(cudaMemset(In.data(), 0, size * sizeof(float)));
  }

}

__global__
void gMapMatrix(MatrixWrapper<float> inWrap,
                const MatrixWrapper<int> mappingWrap,
                int mappingCols, int i)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < inWrap.size()) {
    int numCols = inWrap.dim(1);
    int batchIdx = tid / numCols;
    int col = tid % numCols;

    //inWrap[tid] *= mappingWrap(i, batchIdx, 0, 0);
    inWrap(batchIdx, col, 0, 0) *= mappingWrap(i, batchIdx, 0, 0); // [mappingCols * batchIdx + i];
  }
}

void MapMatrix(Matrix& state, const DeviceVector<int>& mapping, size_t i)
{
  // blank out rows in the state matrix where the word position i does not exist
  // mapping is a concatenated array of 1 & 0 of each sentence in the batch to say whether word exists or not.

  int batchSize = state.dim(0);
  int stateLength = state.dim(1);
  int sentenceLength = mapping.size() / batchSize;

  int numThreads = std::min((int)state.size(), MAX_THREADS);
  int numBlocks = (state.size() / numThreads) + 1;

  MatrixWrapper<float> stateWrap(state);
  MatrixWrapper<int> mappingWrap(mapping, sentenceLength, batchSize, 1, 1);

  gMapMatrix<<<numBlocks, numThreads, 0, CudaStreamHandler::GetStream()>>>
    (stateWrap, mappingWrap, sentenceLength, i);

  /*
  cerr << "nBlocks=" << numBlocks << endl;
  cerr << "nThreads=" << numThreads << endl;
  cerr << "stateWrap=" << stateWrap.Debug() << endl;
  cerr << "mapping=" << Debug(mapping, 2) << endl;
  cerr << "i=" << i << endl;
  cerr << std::endl;

  HANDLE_ERROR(cudaDeviceSynchronize());
  */
}

__global__ void gLNormalization(MatrixWrapper<float> outWrap,
                                const MatrixWrapper<float> inWrap,
                                const MatrixWrapper<float> alphaWrap,
                                const MatrixWrapper<float> betaWrap,
                                float* out, const float* in, const float* alpha, const float* beta,
                                int rows, int cols, float eps=0.00001)
{
  extern __shared__ float _share[];

  for (int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if (j < rows) {
      float* so = out + j * cols;
      const float* sp = in + j * cols;

      float* _sum = _share + blockDim.x;
      _sum[threadIdx.x] = 0.0f;
      for (int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if (id < cols) {
          _sum[threadIdx.x] += sp[id];
        }
      }
      __syncthreads();
      int len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if (threadIdx.x < (len >> 1)) {
          _sum[threadIdx.x] += _sum[threadIdx.x + skip];
        }
        len = (len + 1) >> 1;
      }
      __syncthreads();
      float mean = _sum[0] / cols;
      __syncthreads();

      float* _sqSum = _share + blockDim.x;

      _sqSum[threadIdx.x] = 0.0;
      for (int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          float ex = sp[id] - mean;
          so[id] = ex;
          _sqSum[threadIdx.x] += ex * ex;
        }
      }
      __syncthreads();
      len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1))
          _sqSum[threadIdx.x] += _sqSum[threadIdx.x + skip];
        len = (len + 1) >> 1;
      }
      __syncthreads();
      float sigma = sqrtf(eps + (_sqSum[0] / cols));
      __syncthreads();

      for (int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          if (beta != nullptr) {
            so[id] = alpha[id] * (so[id] / sigma) + beta[id];
          } else {
            so[id] = alpha[id] * (so[id] / sigma);
          }
        }
      }
    }
  }
}

void Normalization(Matrix& out, const Matrix& in, const Matrix& alpha, const Matrix& beta,
                       float eps) {
  int numThreads = std::min((int)in.dim(1), 512);

  out.Reshape(in.dim(0), in.dim(1), 1, 1);

  int rows = in.dim(0);
  int cols = in.dim(1);
  int numBlocks = std::min(rows, 65000);
  int shared = numThreads * sizeof(float) * 2;

  MatrixWrapper<float> outWrap(out);
  const MatrixWrapper<float> inWrap(in);
  const MatrixWrapper<float> alphaWrap(alpha);
  const MatrixWrapper<float> betaWrap(beta);

  gLNormalization<<<numBlocks, numThreads, shared, CudaStreamHandler::GetStream()>>>
    (outWrap, inWrap, alphaWrap, betaWrap, out.data(), in.data(), alpha.data(), beta.data(), rows, cols, eps);
}

void Normalization(Matrix& out, const Matrix& in, const Matrix& alpha, float eps)
{
  int numThreads = std::min((int)in.dim(1), 512);

  out.Reshape(in.dim(0), in.dim(1), 1, 1);

  int rows = in.dim(0);
  int cols = in.dim(1);
  int numBlocks = std::min(rows, 65000);
  int shared = numThreads * sizeof(float) * 2;

  MatrixWrapper<float> outWrap(out);
  const MatrixWrapper<float> inWrap(in);
  const MatrixWrapper<float> alphaWrap(alpha);
  const MatrixWrapper<float> betaWrap;

  gLNormalization<<<numBlocks, numThreads, shared, CudaStreamHandler::GetStream()>>>
    (outWrap, inWrap, alphaWrap, betaWrap, out.data(), in.data(), alpha.data(), nullptr, rows, cols, eps);
}

}  // namespace mblas
}  // namespace GPU
}  // namespace amunmt
