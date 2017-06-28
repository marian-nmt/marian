#include "gpu/mblas/matrix_functions.h"
#include "gpu/mblas/handles.h"

using namespace std;

namespace amunmt {
namespace GPU {
namespace mblas {

thread_local CudaStreamHandler CudaStreamHandler::instance_;
thread_local CublasHandler CublasHandler::instance_;

Matrix& Swap(Matrix& Out, Matrix& In) {
  Out.swap(In);
  return Out;
}

__global__ void gMean(MatrixWrapper<float> out,
                      const MatrixWrapper<float> in,
                      const MatrixWrapper<uint>  mapping)
{
  // out = batches * states
  // in = max sentence length * states * 1 * batches
  // mapping = max length * batches

  int id = threadIdx.x + blockIdx.x * blockDim.x;
  //printf("id = %d in = %lu %lu %lu %lu = %lu %lu \n", id, in.dim(0), in.dim(1), in.dim(2), in.dim(3), in.size(), sizeof(in));

  if (id < out.size()) {
    uint indices[SHAPE_SIZE];
    out.id2Indices(id, indices);
    //printf("%d -> %lu %lu %lu %lu \n", id, indices[0], indices[1], indices[2], indices[3]);

    size_t batch = indices[0];
    size_t state = indices[1];

    float sum = 0.0f;
    int counter = 0;
    for (size_t row = 0; row < in.dim(0); ++row) {
      int isWord = mapping(row, batch, 0, 0);
      //printf("batch=%lu startMapInd=%lu  mapOffset=%lu -> %d \n", batch, startMapInd, mapOffset, isWord);
      if (isWord) {
        sum += in(row, state, 0, batch);
        ++counter;
      }
    }

    sum /= (float) counter;
    out[id] = sum;
  }
}

void Mean(Matrix& Out, const Matrix& In, const IMatrix &sentencesMask)
{
  assert(Out.dim(2) == 1);
  assert(Out.dim(3) == 1);
  assert(Out.dim(0) == In.dim(3));
  assert(Out.dim(1) == In.dim(1));
  assert(In.dim(0) * In.dim(3) == sentencesMask.size());

  // mean of each ROW
  size_t batchNum = Out.dim(0) * Out.dim(2) * Out.dim(3);
  size_t stateLength = Out.dim(1);
  size_t sentenceLength = (In.dim(0) * In.dim(2) * In.dim(3)) / batchNum;

  MatrixWrapper<float> outWrap(Out);
  MatrixWrapper<float> inWrap(In);

  MatrixWrapper<uint> mappingWrap(sentencesMask, false);

  size_t threads = MAX_THREADS;
  size_t blocks =  (outWrap.size() / threads) + ((outWrap.size() % threads == 0) ?  0 : 1);

  gMean<<<blocks, threads, 0, CudaStreamHandler::GetStream()>>>
    (outWrap, inWrap, mappingWrap);

}

__global__ void gWeightedMean(MatrixWrapper<float> out,
                              const MatrixWrapper<float> weights,
                              const MatrixWrapper<float> in,
                              const MatrixWrapper<uint> mapping
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

void WeightedMean(Matrix& Out,const Matrix& Weights, const Matrix& In, const DeviceVector<uint>& mapping) {
  int numHypos = Weights.dim(0);
  int states = In.dim(1);

  Out.NewSize(numHypos, states);

  MatrixWrapper<float> outWrap(Out);
  MatrixWrapper<float> weightsWrap(Weights);
  MatrixWrapper<float> inWrap(In);
  MatrixWrapper<uint> mappingWrap(mapping);

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

  Out.NewSize(n, m);

  float alpha = 1.0;
  float beta  = 0.0;

  cublasSgeam(CublasHandler::GetHandle(), CUBLAS_OP_T, CUBLAS_OP_T, m, n, &alpha, In.data(), n,
              &beta, In.data(), n, Out.data(), m);

  return Out;
}

Matrix& Transpose(Matrix& Out) {
  thread_local Matrix Temp;
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
  Out.NewSize(In.dim(0), In.dim(1), In.dim(2), In.dim(3));

  mblas::copy(In.data(), In.size(), Out.data(), cudaMemcpyDeviceToDevice);

  return Out;
}

__global__ void gPasteRows(  MatrixWrapper<float> out,
                          const MatrixWrapper<float> in,
                          int rowNo, int colNo)
{
  int inRows = in.dim(0);
  int inCols = in.dim(1);

  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id < inRows * inCols) {
    int outCols = out.dim(1);

    int inRow = id / inCols;
    int inCol = id % inCols;

    //out[outID] = in[id];
    out(rowNo, inCol + colNo, 0, inRow) = in(inRow, inCol, 0, 0);
  }
}

void PasteRows(Matrix& Out, const Matrix& In, const size_t rowNo, size_t colNo)
{
  MatrixWrapper<float> outWrap(Out);
  MatrixWrapper<float> inWrap(In);

  int nThreads = MAX_THREADS;
  int nBlocks =  (In.size() / MAX_THREADS) + ((In.size() % MAX_THREADS == 0) ?  0 : 1);

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
  Out.NewSize(1, length);
  size_t start = r * In.dim(1) + c;
  //size_t end   = start + length;

  //mblas::copy(In.begin() + start, In.begin() + end, Out.begin());
  mblas::copy(In.data() + start, length , Out.data(), cudaMemcpyDeviceToDevice);

  return Out;
}

__global__ void gCopyRows(MatrixWrapper<float> out,
                          const MatrixWrapper<float> in,
                          const MatrixWrapper<uint> indicesWrap)
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;

  if (id < out.size()) {
	  uint dim[SHAPE_SIZE];
	  out.id2Indices(id, dim);

	  size_t indicesInd = dim[0];
	  size_t inRow =indicesWrap[indicesInd];

      out(indicesInd, dim[1], 0, 0) = in(inRow, dim[1], 0, 0);

  }
}

Matrix& CopyRows(Matrix& Out,
                 const Matrix& In,
                 const DeviceVector<uint>& indices)
{
  assert(In.dim(1) == Out.dim(1));
  assert(Out.dim(0) == indices.size());

  assert(In.dim(2) == 1);
  assert(In.dim(3) == 1);
  assert(Out.dim(2) == 1);
  assert(Out.dim(3) == 1);

  /*
  cerr << "Out=" << Out.Debug(0) << endl;
  cerr << "In=" << In.Debug(0) << endl;
  cerr << "indices=" << Debug(indices, 2) << endl;
  cerr << endl;
  */

  size_t size = Out.size();

  size_t numPairs = indices.size();

  MatrixWrapper<float> outWrap(Out);
  const MatrixWrapper<float> inWrap(In);
  const MatrixWrapper<uint> indicesWrap(indices);

  uint threads = std::min((uint) MAX_THREADS, (uint)size);
  int blocks = size / threads + 1;

  gCopyRows<<<blocks, threads, 0, CudaStreamHandler::GetStream()>>>
    (outWrap, inWrap, indicesWrap);

  return Out;
}


Matrix& Assemble(Matrix& Out,
                 const Matrix& In,
                 const DeviceVector<uint>& indices) {
  Out.NewSize(indices.size(), In.dim(1));
  //cerr << "Assemble=" << Out.Debug() << " " << In.Debug() << indices.size() << endl;

  CopyRows(Out, In, indices);
  return Out;
}

__global__ void gSlice(MatrixWrapper<float> out,
                      const MatrixWrapper<float> in,
                       size_t n, size_t dim)
{
  size_t row = blockIdx.x;

  size_t inCol = threadIdx.x + dim * n;
  size_t outCol = threadIdx.x;

  while (outCol < out.dim(1)) {
    out(row, outCol, 0, 0) = in(row, inCol, 0, 0);

    inCol += blockDim.x;
    outCol += blockDim.x;
  }

}

Matrix& Slice(Matrix& Out,
              const Matrix& In,
              size_t n, size_t dim)
{
  assert(In.dim(2) == 1);
  assert(In.dim(3) == 1);

  Out.NewSize(In.dim(0), dim);

  MatrixWrapper<float> outWrap(Out);
  const MatrixWrapper<float> inWrap(In);

  /*
  cerr << "outWrap=" << outWrap.Debug() << endl;
  cerr << "inWrap=" << inWrap.Debug() << endl;
  cerr << "n=" << n << endl;
  cerr << "dim=" << dim << endl;
  cerr << endl;
  */

  uint threads = std::min((uint)MAX_THREADS, (uint)dim);
  uint blocks = In.dim(0);

  gSlice<<<blocks, threads, 0, CudaStreamHandler::GetStream()>>>
    (outWrap, inWrap, n, dim);
  return Out;
}

Matrix& Prod(cublasHandle_t handle, Matrix& C, const Matrix& A, const Matrix& B,
             bool transA, bool transB)
{
  assert((A.dim(2) == A.dim(3) == 1) || (B.dim(2) == B.dim(3) == 1));

  Matrix::value_type alpha = 1.0;
  Matrix::value_type beta = 0.0;

  size_t m = A.dim(0) * A.dim(2) * A.dim(3);
  size_t k = A.dim(1);
  size_t mOut = A.dim(0);
  size_t kOut = A.dim(1);
  if(transA) {
    std::swap(m, k);
    std::swap(mOut, kOut);
  }

  size_t l = B.dim(0) * B.dim(2) * B.dim(3);
  size_t n = B.dim(1);
  size_t lOut = B.dim(0);
  size_t nOut = B.dim(1);
  if(transB) {
    std::swap(l, n);
    std::swap(lOut, nOut);
    }

  assert(k == l);

  size_t lda = A.dim(1);
  size_t ldb = B.dim(1);
  size_t ldc = transB ? B.dim(0) * B.dim(2) * B.dim(3) : B.dim(1);

  size_t dim2 = A.dim(2);
  if (!transA && transB) {
    // for GetAlignedSourceContext()
    assert((A.dim(2) == A.dim(3) == 1));
    C.NewSize(nOut, B.dim(2), 1, 1);
  }
  else {
    C.NewSize(mOut, nOut, A.dim(2) * B.dim(2), A.dim(3) * B.dim(3));
  }

  /*
  cerr << "C=" << C.Debug(0) << endl;
  cerr << "A=" << A.Debug(0) << endl;
  cerr << "B=" << B.Debug(0) << endl;
  cerr << "transA=" << transA << endl;
  cerr << "transB=" << transB << endl;
  cerr << endl;
  */
  cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

  /*
   cublasStatus_t cublasSgemm(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const float           *alpha,
                           const float           *A, int lda,
                           const float           *B, int ldb,
                           const float           *beta,
                           float           *C, int ldc)
   */
  cublasSgemm(handle, opB, opA,
              n, m, k,
              &alpha,
              B.data(), ldb,
              A.data(), lda,
              &beta,
              C.data(), ldc);
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

__global__ void gSoftMax(MatrixWrapper<float> out,
                         const MatrixWrapper<uint> batchIdsWrap,
                         const MatrixWrapper<uint> sentencesMappingWrap,
                         uint shareSize)
{
  extern __shared__ float _share[];

  size_t numHypos = out.dim(0);
  size_t srcLen = out.dim(1);

  int hypoInd =  blockIdx.x;
  int origSrcPos = threadIdx.x;

  while (hypoInd < numHypos) {
    MatrixWrapper<float> _max(_share, shareSize);
    _max[origSrcPos] = out(hypoInd, origSrcPos, 0, 0);
    for (int tid = 0; tid < srcLen; tid += blockDim.x) {
      int srcPos = tid + origSrcPos;
      if (srcPos < srcLen) {
        float value = out(hypoInd, srcPos, 0, 0);

        int batch = batchIdsWrap[hypoInd];
        value *= sentencesMappingWrap(srcPos, batch, 0, 0);
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

    //float* _sum = _share;// + blockDim.x;
    MatrixWrapper<float> _sum(_share, shareSize);

    _sum[origSrcPos] = 0.0f;
    for (int tid = 0; tid < srcLen; tid += blockDim.x) {
      int srcPos = tid + origSrcPos;
      if (srcPos < srcLen) {
        out(hypoInd, srcPos, 0, 0) = __expf(out(hypoInd, srcPos, 0, 0) - max);

        int batch = batchIdsWrap[hypoInd];
        out(hypoInd, srcPos, 0, 0) *= sentencesMappingWrap(srcPos, batch, 0, 0);
        _sum[origSrcPos] += out(hypoInd, srcPos, 0, 0);
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
        out(hypoInd, srcPos, 0, 0) /= _sum[0];
      }
    }
    __syncthreads();
    hypoInd += gridDim.x;
  }
}

Matrix& Softmax(Matrix& Out, const DeviceVector<uint>& batchIds, const mblas::IMatrix &sentencesMask, size_t batchSize)
{
  size_t srcSize = Out.dim(1);

  MatrixWrapper<float> outWrap(Out);
  const MatrixWrapper<uint> batchIdsWrap(batchIds);
  const MatrixWrapper<uint> sentencesMappingWrap(sentencesMask, false);

  int blocks = batchSize;
  int threads = std::min(MAX_THREADS, (int)srcSize);
  int shared = sizeof(float) * threads;

  gSoftMax<<<blocks, threads, shared, CudaStreamHandler::GetStream()>>>
    (outWrap, batchIdsWrap, sentencesMappingWrap, threads);

  return Out;
}

__global__ void gLogSoftMax(MatrixWrapper<float> out, uint shareSize)
{
  extern __shared__ float _share[];

  size_t rows = out.dim(0);
  size_t cols = out.dim(1);

  int rowIdx =  blockIdx.x;

  while (rowIdx < rows) {
    //float* _max = _share;
    MatrixWrapper<float> _max(_share, shareSize);

    _max[threadIdx.x] = out(rowIdx, threadIdx.x, 0, 0);
    for (int tid = 0; tid < cols; tid += blockDim.x) {
      int id = tid + threadIdx.x;
      if (id < cols) {
        const float &val = out(rowIdx, id, 0, 0);
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

    //float* _sum = _share;// + blockDim.x;
    MatrixWrapper<float> _sum(_share, shareSize);

    _sum[threadIdx.x] = 0.0f;
    for (int tid = 0; tid < cols; tid += blockDim.x) {
      int id = tid + threadIdx.x;
      if (id < cols) {
        //row[id] = exp(row[id] - max);
        float &val = out(rowIdx, id, 0, 0);
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
        float &val = out(rowIdx, id, 0, 0);
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
    (Out, threads);

  return Out;
}

__global__ void gSetColumn(MatrixWrapper<float> in, int noColumn, float value) {
  int n_rows = in.dim(0);

  int rowNumber = threadIdx.x  + blockDim.x * blockIdx.x;

  if (rowNumber < n_rows) {
    in(rowNumber, noColumn, 0, 0) = value;
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

__global__ void gFill(MatrixWrapper<float> in, float val) {
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  if (index < in.size()) {
    in[index] = val;
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
    HANDLE_ERROR(cudaMemsetAsync(In.data(), 0, size * sizeof(float), CudaStreamHandler::GetStream()));
  }

}

__global__
void gMapMatrix(MatrixWrapper<float> in,
                const MatrixWrapper<uint> sentencesMappingWrap,
                int mappingCols, int i)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < in.size()) {
    int numCols = in.dim(1);
    int batchIdx = tid / numCols;
    int col = tid % numCols;

    //in[tid] *= mappingWrap(i, batchIdx, 0, 0);
    in(batchIdx, col, 0, 0) *= sentencesMappingWrap(i, batchIdx, 0, 0); // [mappingCols * batchIdx + i];
  }
}

void MapMatrix(Matrix& state, const mblas::IMatrix &sentencesMask, size_t i)
{
  // blank out rows in the state matrix where the word position i does not exist
  // mapping is a concatenated array of 1 & 0 of each sentence in the batch to say whether word exists or not.

  int batchSize = state.dim(0);
  int stateLength = state.dim(1);
  int sentenceLength = sentencesMask.size() / batchSize;

  int numThreads = std::min((int)state.size(), MAX_THREADS);
  int numBlocks = (state.size() / numThreads) + 1;

  MatrixWrapper<float> stateWrap(state);
  MatrixWrapper<uint> sentencesMappingWrap(sentencesMask, false);

  gMapMatrix<<<numBlocks, numThreads, 0, CudaStreamHandler::GetStream()>>>
    (stateWrap, sentencesMappingWrap, sentenceLength, i);

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

__device__ uint getIndex(const dim3 &dim, const dim3 &val)
{
  uint ret = dim.x * val.x + dim.y * val.y + dim.z * val.z;
  return ret;
}


__global__ void gLNormalization(MatrixWrapper<float> out,
                                const MatrixWrapper<float> in,
                                const MatrixWrapper<float> alphaWrap,
                                const MatrixWrapper<float> betaWrap,
                                float eps=0.00001)
{
  extern __shared__ float _share[];

  //printf("blockDim.x=%d gridDim.x=%d \n", blockDim.x, gridDim.x);
  // blockDim.x=512 gridDim.x=1

  int cols = in.dim(1);

  assert(blockIdx.x < in.dim(0));
  assert(blockIdx.y < in.dim(2));
  assert(blockIdx.z < in.dim(3));

  float* _sum = _share + blockDim.x;
  _sum[threadIdx.x] = 0.0f;
  for (int tid = 0; tid < cols; tid += blockDim.x) {
    int id = tid + threadIdx.x;
    if (id < cols) {
      _sum[threadIdx.x] += in(blockIdx.x, id, blockIdx.y, blockIdx.z);
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
      float ex = in(blockIdx.x, id, blockIdx.y, blockIdx.z) - mean;
      out(blockIdx.x, id, blockIdx.y, blockIdx.z) = ex;
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
      float &val = out(blockIdx.x, id, blockIdx.y, blockIdx.z);
      if (betaWrap.size()) {
        val = alphaWrap[id] * (val / sigma) + betaWrap[id];
      } else {
        val = alphaWrap[id] * (val / sigma);
      }
    }
  }

}

void Normalization(Matrix &out,
                  const Matrix &in,
                  const Matrix &alpha,
                  const Matrix *beta,
                  float eps)
{
  assert(in.dim(0) < MAX_BLOCKS);
  assert(in.dim(2) < MAX_BLOCKS);
  assert(in.dim(3) < MAX_BLOCKS);

  //out.Reshape(in.dim(0), in.dim(1), in.dim(2), in.dim(3));

  int numThreads = std::min((uint) in.dim(1), (uint) MAX_THREADS);
  dim3 numBlocks(in.dim(0), in.dim(2), in.dim(3));
  int shared = numThreads * sizeof(float) * 2;

  MatrixWrapper<float> outWrap(out);
  const MatrixWrapper<float> inWrap(in);
  const MatrixWrapper<float> alphaWrap(alpha);
  MatrixWrapper<float> *betaWrap = beta ? new MatrixWrapper<float>(*beta) : new MatrixWrapper<float>();

  gLNormalization<<<numBlocks, numThreads, shared, CudaStreamHandler::GetStream()>>>
    (outWrap, inWrap, alphaWrap, *betaWrap, eps);

  /*
  //std::cerr << "nBlocks=" << numBlocks << std::endl;
  std::cerr << "nThreads=" << numThreads << std::endl;
  std::cerr << "outWrap=" << outWrap.Debug() << std::endl;
  std::cerr << "inWrap=" << inWrap.Debug() << std::endl;
  std::cerr << "alphaWrap=" << alphaWrap.Debug() << std::endl;
  std::cerr << "betaWrap=" << betaWrap->Debug() << std::endl;
  std::cerr << std::endl;

  HANDLE_ERROR(cudaDeviceSynchronize());
  */
  delete betaWrap;
}

void Normalization(Matrix& out, const Matrix& in, const Matrix& alpha, const Matrix& beta,
                       float eps)
{
  Normalization(out, in, alpha, &beta, eps);
}

void Normalization(Matrix& out, const Matrix& in, const Matrix& alpha, float eps)
{
  Normalization(out, in, alpha, nullptr, eps);
}

__global__ void gRandomizeMemory(int *data)
{
  clock_t start = clock();

}

void RandomizeMemory()
{
  int *data;
  HANDLE_ERROR( cudaMalloc((void**)&data, 8 * 1024 ^ 3) );

  uint threads = 1024;
  uint blocks = 8 * 1024 ^ 3 / threads;
  gRandomizeMemory<<<blocks, threads>>>(data);
}

}  // namespace mblas
}  // namespace GPU
}  // namespace amunmt
