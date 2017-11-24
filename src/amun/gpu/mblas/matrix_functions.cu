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
                      const MatrixWrapper<uint> sentenceLengths)
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
      bool isWord = row < sentenceLengths[batch];
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

void Mean(Matrix& Out,
          const Matrix& In,
          const mblas::IMatrix &sentenceLengths)
{
  assert(Out.dim(2) == 1);
  assert(Out.dim(3) == 1);
  assert(Out.dim(0) == In.dim(3));
  assert(Out.dim(1) == In.dim(1));

  // mean of each ROW
  size_t batchNum = Out.dim(0) * Out.dim(2) * Out.dim(3);
  size_t stateLength = Out.dim(1);
  size_t sentenceLength = (In.dim(0) * In.dim(2) * In.dim(3)) / batchNum;

  MatrixWrapper<float> outWrap(Out);
  MatrixWrapper<float> inWrap(In);
  //cerr << "outWrap=" << outWrap.Debug() << endl;

  MatrixWrapper<uint> sentenceLengthsWrap(sentenceLengths, false);

  uint size = outWrap.size();
  uint threads = std::min((uint)MAX_THREADS, size);
  uint blocks =  (size / threads) + ((size % threads == 0) ?  0 : 1);

  gMean<<<blocks, threads, 0, CudaStreamHandler::GetStream()>>>
    (outWrap, inWrap, sentenceLengthsWrap);

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

void WeightedMean(Matrix& Out,const Matrix& Weights, const Matrix& In, const mblas::Array<uint>& mapping)
{
  int numHypos = Weights.dim(0);
  int states = In.dim(1);

  Out.NewSize(numHypos, states);

  MatrixWrapper<float> outWrap(Out);
  MatrixWrapper<float> weightsWrap(Weights);
  MatrixWrapper<float> inWrap(In);
  MatrixWrapper<uint> mappingWrap(mapping);

  uint size = Out.size();
  uint nThreads = std::min((uint) MAX_THREADS, (uint)size);
  uint nBlocks =  (size / nThreads) + ((size % nThreads == 0) ?  0 : 1);

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

  uint size = In.size();
  uint nThreads = std::min((uint) MAX_THREADS, (uint)size);
  uint nBlocks =  (size / nThreads) + ((size % nThreads == 0) ?  0 : 1);

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
                 const mblas::Array<uint>& indices)
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
  //cerr << "size=" << size << endl;

  uint threads = std::min((uint) MAX_THREADS, (uint)size);
  uint blocks = size / threads + ((size % threads == 0) ?  0 : 1);

  gCopyRows<<<blocks, threads, 0, CudaStreamHandler::GetStream()>>>
    (outWrap, inWrap, indicesWrap);

  return Out;
}


Matrix& Assemble(Matrix& Out,
                 const Matrix& In,
                 const mblas::Array<uint>& indices) {
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
                         const MatrixWrapper<uint> sentenceLengthsWrap,
                         uint shareSize)
{
  extern __shared__ float _share[];

  size_t numHypos = out.dim(0);
  size_t maxLength = out.dim(1);

  int hypoInd =  blockIdx.x;
  int origSrcPos = threadIdx.x;

  while (hypoInd < numHypos) {
    MatrixWrapper<float> _max(_share, shareSize, 1, 1, 1);
    _max[origSrcPos] = out(hypoInd, origSrcPos, 0, 0);
    for (int tid = 0; tid < maxLength; tid += blockDim.x) {
      int srcPos = tid + origSrcPos;
      if (srcPos < maxLength) {
        float value = out(hypoInd, srcPos, 0, 0);

        int batch = batchIdsWrap[hypoInd];
        value *= srcPos < sentenceLengthsWrap[batch] ? 1 : 0;
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
    MatrixWrapper<float> _sum(_share, shareSize, 1, 1, 1);

    _sum[origSrcPos] = 0.0f;
    for (int tid = 0; tid < maxLength; tid += blockDim.x) {
      int srcPos = tid + origSrcPos;
      if (srcPos < maxLength) {
        out(hypoInd, srcPos, 0, 0) = __expf(out(hypoInd, srcPos, 0, 0) - max);

        int batch = batchIdsWrap[hypoInd];
        out(hypoInd, srcPos, 0, 0) *= srcPos < sentenceLengthsWrap[batch] ? 1 : 0; // sentencesMappingWrap(srcPos, batch, 0, 0);
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

    for (int tid = 0; tid < maxLength; tid += blockDim.x) {
      int srcPos = tid + origSrcPos;
      if (srcPos < maxLength) {
        out(hypoInd, srcPos, 0, 0) /= _sum[0];
      }
    }
    __syncthreads();
    hypoInd += gridDim.x;
  }
}

Matrix& Softmax(Matrix& Out,
                const mblas::Array<uint>& batchIds,
                const mblas::IMatrix &sentenceLengths,
                size_t batchSize)
{
  size_t maxLength = Out.dim(1);

  MatrixWrapper<float> outWrap(Out);
  const MatrixWrapper<uint> batchIdsWrap(batchIds);
  const MatrixWrapper<uint> sentenceLengthsWrap(sentenceLengths, false);

  int blocks = batchSize;
  int threads = std::min(MAX_THREADS, (int)maxLength);
  int shared = sizeof(float) * threads;

  gSoftMax<<<blocks, threads, shared, CudaStreamHandler::GetStream()>>>
    (outWrap, batchIdsWrap, sentenceLengthsWrap, threads);

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
    MatrixWrapper<float> _max(_share, shareSize, 1, 1, 1);

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
    MatrixWrapper<float> _sum(_share, shareSize, 1, 1, 1);

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
                const MatrixWrapper<uint> sentenceLengthsWrap,
                int i)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < in.size()) {
    int numCols = in.dim(1);
    int batchIdx = tid / numCols;
    int col = tid % numCols;

    //in[tid] *= mappingWrap(i, batchIdx, 0, 0);
    in(batchIdx, col, 0, 0) *= (i < sentenceLengthsWrap[batchIdx] ? 1 : 0);
  }
}

void MapMatrix(Matrix& state,
              const mblas::IMatrix &sentenceLengths,
              size_t i)
{
  // blank out rows in the state matrix where the word position i does not exist
  // mapping is a concatenated array of 1 & 0 of each sentence in the batch to say whether word exists or not.

  int batchSize = state.dim(0);
  int stateLength = state.dim(1);

  int numThreads = std::min((int)state.size(), MAX_THREADS);
  int numBlocks = (state.size() / numThreads) + ((state.size() % numThreads == 0) ? 0 : 1);

  MatrixWrapper<float> stateWrap(state);
  MatrixWrapper<uint> sentenceLengthsWrap(sentenceLengths);

  gMapMatrix<<<numBlocks, numThreads, 0, CudaStreamHandler::GetStream()>>>
    (stateWrap, sentenceLengthsWrap, i);

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

///////////////////////////////////////////////////////////////////////////////////////////////////////
__global__
void gBeamSizeInit(MatrixWrapper<uint> hypo2BeamSizeWrap,
                  MatrixWrapper<uint> batch2HypoWrap,
                  MatrixWrapper<uint> hypo2CandidateWrap,
                  bool isFirst,
                  uint beamSizeSum,
                  const MatrixWrapper<uint> beamSizesWrap)
{
  uint hypoInd = 0;
  uint candidateInd = 0;

  uint a = 0, b = 0;
  //printf("beamSizesWrap.size()=%u \n", beamSizesWrap.size());
  for (size_t batchInd = 0; batchInd < beamSizesWrap.size(); ++batchInd) {
    uint beamSize = beamSizesWrap[batchInd];
    /*
    printf("batchInd=%u ", batchInd);
    printf("beamSize=%u ", beamSize);
    printf("a=%u ", a);
    printf("b=%u \n", b);
    */

    if (beamSize) {
      if (isFirst) {
        assert(a < hypo2BeamSizeWrap.size());
        assert(a < hypo2CandidateWrap.size());
        hypo2BeamSizeWrap[a] = beamSize;
        hypo2CandidateWrap[a] = candidateInd;
        ++a;

        assert(b < batch2HypoWrap.size());
        batch2HypoWrap[b] = batchInd;
        ++b;

        candidateInd += beamSize;
      }
      else {
        for (size_t j = 0; j < beamSize; ++j) {
          assert(a < hypo2BeamSizeWrap.size());
          assert(a < hypo2CandidateWrap.size());
          hypo2BeamSizeWrap[a] = beamSize;
          hypo2CandidateWrap[a] = candidateInd;
          ++a;

          candidateInd += beamSize;
        }

        assert(b < batch2HypoWrap.size());
        batch2HypoWrap[b] = hypoInd;
        ++b;
      }

      hypoInd += beamSize;
    }
  }

}

__device__
float GetMaxScore(const MatrixWrapper<NthOutBatch> &nBestMatrix)
{
  float ret = -1111111111111;
  for (uint i = 0; i < nBestMatrix.dim(1); ++i) {
      const NthOutBatch &curr = nBestMatrix[i];
      if (curr.score > ret) {
        ret = curr.score;
    }
  }

  return ret;
}

__device__
void AddElement(float &minScore,
    uint &i,
    NthOutBatch *arr,
    bool forbidUNK,
    uint vocabInd,
    const NthOutBatch &ele)
{
  const float score = ele.score;

  if (forbidUNK && vocabInd == UNK_ID) {
    arr[i].score = -1111111111111;
    minScore = -1111111111111;
  }
  else {
    arr[i] = ele;

    if (score < minScore) {
      minScore = score;
    }

    ++i;
  }

}

__device__
void MergeElement(float &minScore,
                  NthOutBatch *arr,
                  uint arrSize,
                  const NthOutBatch &ele)
{
  float newMinScore = +1111111111;
  bool found = false;
  for (uint i = 0; i < arrSize; ++i) {
    NthOutBatch &currEle = arr[i];
    if (!found && minScore == currEle.score) {
      currEle = ele;
      found = true;
    }

    // update min score
    if (currEle.score < newMinScore) {
      newMinScore = currEle.score;
    }
  }

  minScore = newMinScore;
}

__device__
void MergeElement(float &minScore,
                  NthOutBatch *arr,
                  uint arrSize,
                  const NthOutBatch &ele,
                  bool forbidUNK,
                  uint vocabInd)
{
  if (forbidUNK && vocabInd == UNK_ID) {
    // do nothing
  }
  else if (ele.score > minScore) {
    // replace element with min score
    MergeElement(minScore, arr, arrSize, ele);

    /*
    printf("arrInd=%d ind=%d vocabId=%d \n",
          arrInd,
          _max[threadIdx.x].ind,
          _max[threadIdx.x].vocabId);
    */
  }
}

__device__
void NBestAndMax(MatrixWrapper<NthOutBatch> nBestCandidatesWrap,
              float &topScore,
              const MatrixWrapper<float> in,
              const MatrixWrapper<float> b4Wrap,
              uint hypoInd,
              uint maxBeamSize,
              bool forbidUNK,
              const MatrixWrapper<uint> hypo2BeamSizeWrap,
              const MatrixWrapper<uint> hypo2CandidateWrap)
{
  extern __shared__ char _sharePtr[];

  MatrixWrapper<float> maxMatrix((float*)_sharePtr, blockDim.x, 1, 1, 1);

  void *ptrOffset = _sharePtr + sizeof(float) * blockDim.x;
  MatrixWrapper<NthOutBatch> nBestMatrix((NthOutBatch*)ptrOffset, blockDim.x, maxBeamSize, 1, 1);
  NthOutBatch *arr = &nBestMatrix(threadIdx.x, 0, 0, 0);

  uint vocabSize = in.dim(1);

  assert(hypoInd < hypo2BeamSizeWrap.size());
  uint beamSize = hypo2BeamSizeWrap[hypoInd];

  float minScore = +1111111111;

  // init
  uint vocabInd = threadIdx.x;
  uint i = 0;
  while (vocabInd < vocabSize && i < beamSize) {
    const float score = in(hypoInd, vocabInd, 0, 0) + b4Wrap(0, vocabInd, 0, 0);

    uint arrInd = hypoInd * vocabSize + vocabInd;
    NthOutBatch ele(arrInd, score, hypoInd, vocabInd);

    AddElement(minScore, i, arr, forbidUNK, vocabInd, ele);

    vocabInd += blockDim.x;
  }

  // MAIN LOOP
  while (vocabInd < vocabSize) {
    const float score = in(hypoInd, vocabInd, 0, 0) + b4Wrap(0, vocabInd, 0, 0);
    uint arrInd = hypoInd * vocabSize + vocabInd;
    NthOutBatch ele(arrInd, score, hypoInd, vocabInd);

    MergeElement(minScore, arr, beamSize, ele, forbidUNK, vocabInd);

    vocabInd += blockDim.x;
  } // while (vocabInd < vocabSize) {

  // merge nbest from different threads
  int len = blockDim.x;
  while (len != 1) {
    __syncthreads();
    int skip = (len + 1) >> 1;
    if (threadIdx.x < (len >> 1)) {
      NthOutBatch *dest = &nBestMatrix(threadIdx.x, 0, 0, 0);

      for (uint i = 0; i < beamSize; ++i) {
        const NthOutBatch &ele = nBestMatrix(threadIdx.x + skip, i, 0, 0);
        if (ele.score > minScore) {
          MergeElement(minScore, dest, beamSize, ele);
        }
      }
    }
    len = (len + 1) >> 1;

  }

  __syncthreads();

  if (threadIdx.x == 0) {
    // copy to output array
    assert(hypoInd < hypo2CandidateWrap.size());
    uint candidateInd = hypo2CandidateWrap[hypoInd];
    for (uint i = 0; i < beamSize; ++i) {
      const NthOutBatch &curr = nBestMatrix(0, i, 0, 0);
      //printf("vocabInd=%u \n", best.vocabInd);

      assert(candidateInd + i < nBestCandidatesWrap.size());
      nBestCandidatesWrap[candidateInd + i] = curr;
    }
  }

  __syncthreads();
  topScore = GetMaxScore(nBestMatrix);

}

///////////////////////////////////////////////////////////////////////////////////////////////////////
__device__
void SumAndLogSoftMax(MatrixWrapper<NthOutBatch> nBestCandidatesWrap,
                            const MatrixWrapper<float> in,
                            const MatrixWrapper<float> b4Wrap,
                            uint hypoInd,
                            uint maxBeamSize,
                            float topScore,
                            const MatrixWrapper<uint> hypo2BeamSizeWrap,
                            const MatrixWrapper<uint> hypo2CandidateWrap)
{
  extern __shared__ float _share[];

  size_t vocabSize = in.dim(1);
  //assert(nBestCandidatesWrap.dim(0) == rows);

  //float* _sum = _share;// + blockDim.x;
  MatrixWrapper<float> _sum(_share, blockDim.x, 1, 1, 1);

  // calc sum
  _sum[threadIdx.x] = 0.0f;
  for (int id = threadIdx.x; id < vocabSize; id += blockDim.x) {
    //row[id] = exp(row[id] - max);
    float val = in(hypoInd, id, 0, 0) + b4Wrap(0, id, 0, 0);
    val = __expf(val - topScore);
    _sum[threadIdx.x] += val;
  }

  int len = blockDim.x;
  while (len != 1) {
    __syncthreads();

    int skip = (len + 1) >> 1;
    if (threadIdx.x < (len >> 1)) {
      _sum[threadIdx.x] += _sum[threadIdx.x + skip];
    }
    len = (len + 1) >> 1;
  }

  __syncthreads();

  // apply partition and log to top
  if (threadIdx.x == 0) {
    //__syncthreads();
    //printf("val=%f %f \n", in(rowIdx, ele.vocabId, 0, 0), val);

    // nbest
    uint beamSize = hypo2BeamSizeWrap[hypoInd];
    uint startPos = hypo2CandidateWrap[hypoInd];
    for (uint i = 0; i < beamSize; ++i) {
      //__syncthreads();
      NthOutBatch &ele = nBestCandidatesWrap[startPos + i];

      float &val = ele.score;
      val = __expf(val - topScore);
      val = __logf(val /_sum[0]);
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void gLogSoftMax(MatrixWrapper<NthOutBatch> nBestCandidatesWrap,
                        const MatrixWrapper<float> in,
                        const MatrixWrapper<float> b4Wrap,
                        uint maxBeamSize,
                        bool forbidUNK,
                        const MatrixWrapper<uint> hypo2BeamSizeWrap,
                        const MatrixWrapper<uint> hypo2CandidateWrap)
{
  uint hypos = in.dim(0);
  uint vocabSize = in.dim(1);

  uint hypoInd =  blockIdx.x; // index of previous hypo
  while (hypoInd < hypos) {
    float topScore;

    NBestAndMax(nBestCandidatesWrap,
            topScore,
            in,
            b4Wrap,
            hypoInd,
            maxBeamSize,
            forbidUNK,
            hypo2BeamSizeWrap,
            hypo2CandidateWrap);

    SumAndLogSoftMax(nBestCandidatesWrap,
                in,
                b4Wrap,
                hypoInd,
                maxBeamSize,
                topScore,
                hypo2BeamSizeWrap,
                hypo2CandidateWrap);


    __syncthreads();
    hypoInd += gridDim.x;
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void gNBestPerBatch(MatrixWrapper<NthOutBatch> nBestWrap,
                        MatrixWrapper<NthOutBatch> nBestCandidatesWrap,
                        const MatrixWrapper<float> in,
                        const MatrixWrapper<float> costsWrap,
                        uint maxBeamSize,
                        bool forbidUNK,
                        bool isFirst,
                        const MatrixWrapper<uint> hypo2BeamSizeWrap,
                        const MatrixWrapper<uint> batch2HypoWrap,
                        const MatrixWrapper<uint> hypo2CandidateWrap)
{
  //uint rows = in.dim(0);
  uint batchSize = batch2HypoWrap.dim(0);

  uint batchInd =  blockIdx.x;
  while (batchInd < batchSize) {
    assert(batchInd < batch2HypoWrap.size());
    assert(batchInd < hypo2BeamSizeWrap.size());
    assert(batchInd < nBestWrap.size());

    uint hypoInd = batch2HypoWrap[batchInd];
    uint beamSize = hypo2BeamSizeWrap[hypoInd];
    assert(beamSize);

    uint nextHypoInd;
    if (isFirst) {
      nextHypoInd = batchInd * beamSize;
    }
    else {
      nextHypoInd = hypoInd;
    }

    // candiate from 1st hypo
    float minScore = +999999;
    assert(hypoInd < hypo2CandidateWrap.size());
    uint candidateInd = hypo2CandidateWrap[hypoInd];
    for (uint i = 0; i < beamSize; ++i) {
      float prevCost;
      if (isFirst) {
        assert(batchInd < costsWrap.size());
        prevCost = costsWrap[batchInd];
      }
      else {
        //printf("prevHypoInd=%, candidateInd=%d \n", prevHypoInd, candidateInd);
        assert(hypoInd < costsWrap.size());
        prevCost = costsWrap[hypoInd];
      }

      assert((nextHypoInd + i) < nBestWrap.size());
      assert(candidateInd + i < nBestCandidatesWrap.size());
      nBestWrap[nextHypoInd + i] = nBestCandidatesWrap[candidateInd + i];

      float &score = nBestWrap[nextHypoInd + i].score;
      score += prevCost;

      if (score < minScore) {
        minScore = score;
      }
    }

    // candidates from other previous hypos
    if (!isFirst) {
      for (uint hypoOffset = 1; hypoOffset < beamSize; ++hypoOffset) {
        //printf("hypoInd=%d \n", (hypoInd + hypoOffset));

        //printf("prevHypoInd=%, candidateInd=%d \n", prevHypoInd, candidateInd);
        assert((hypoInd + hypoOffset) < costsWrap.size());
        float prevCost = costsWrap[hypoInd + hypoOffset];

        assert((hypoInd + hypoOffset) < hypo2CandidateWrap.size());
        uint candidateInd = hypo2CandidateWrap[hypoInd + hypoOffset];

        for (uint candidateOffset = 0; candidateOffset < beamSize; ++candidateOffset) {
          assert((candidateInd + candidateOffset) < nBestCandidatesWrap.size());
          NthOutBatch &candidate = nBestCandidatesWrap[candidateInd + candidateOffset];
          candidate.score += prevCost;

          assert(nextHypoInd < nBestWrap.size());
          NthOutBatch *arr = &nBestWrap[nextHypoInd];

          if (candidate.score > minScore) {
            MergeElement(minScore, arr, beamSize, candidate);
          }
        }
      }
    }

    batchInd += gridDim.x;
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////

void LogSoftmaxAndNBest(mblas::Array<NthOutBatch> &nBest,
                const Matrix& in,
                const Matrix& b4,
                const mblas::Array<float> &costs,
                bool forbidUNK,
                uint maxBeamSize,
                const std::vector<uint>& beamSizes,
                uint beamSizeSum,
                bool isFirst)
{
  //BEGIN_TIMER("LogSoftmax excl kernels");

  //cerr << "in=" << in.Debug(0) << endl;
  //cerr << "beamSizes=" << beamSizes.size() << endl;

  // create beam size vectors on GPU but exclude empty beams
  uint batchSize = 0;
  uint candidateInd = 0;
  for (size_t batchInd = 0; batchInd < beamSizes.size(); ++batchInd) {
    uint beamSize = beamSizes[batchInd];
    //cerr << "(" << beamSize << "," << hypoInd << ") ";

    if (beamSize) {
      if (isFirst) {
        candidateInd += beamSize;
      }
      else {
        candidateInd += beamSize * beamSize;
      }

      ++batchSize;
    }
  }

  mblas::Array<uint> d_beamSizes(beamSizes);
  mblas::Array<uint> hypo2BeamSize(in.dim(0));
  mblas::Array<uint> hypo2Candidate(in.dim(0));
  mblas::Array<uint> batch2Hypo(batchSize);
  mblas::Array<NthOutBatch> nBestCandidates(candidateInd);

  /*
  cerr << "in=" << in.Debug(0) << endl;
  cerr << "beamSizes=" << beamSizes.size() << endl;
  cerr << "beamSizeSum=" << beamSizeSum << endl;
  cerr << "batchSize=" << batchSize << endl;
  cerr << "candidateInd=" << candidateInd << endl;
  cerr << "hypo2BeamSize=" << Debug(hypo2BeamSize, 0) << endl;
  cerr << "hypo2Candidate=" << Debug(hypo2Candidate, 0) << endl;
  cerr << "batch2Hypo=" << Debug(batch2Hypo, 0) << endl;
  cerr << "nBest=" << Debug(nBest, 0) << endl;
  cerr << "nBestCandidates=" << Debug(nBestCandidates, 0) << endl;
  cerr << endl;
  */
  MatrixWrapper<float> inWrap(in);
  MatrixWrapper<float> b4Wrap(b4);
  MatrixWrapper<uint> hypo2BeamSizeWrap(hypo2BeamSize);
  MatrixWrapper<uint> hypo2CandidateWrap(hypo2Candidate);
  MatrixWrapper<uint> batch2HypoWrap(batch2Hypo);
  MatrixWrapper<NthOutBatch> nBestWrap(nBest);
  MatrixWrapper<NthOutBatch> nBestCandidatesWrap(nBestCandidates);
  MatrixWrapper<float> costsWrap(costs);

  MatrixWrapper<uint> beamSizesWrap(d_beamSizes);

  //PAUSE_TIMER("LogSoftmax excl kernels");

  int blocks = std::min(MAX_BLOCKS, (int)in.dim(0));
  int threads = std::min(MAX_THREADS, (int)in.dim(1));
  int shared = sizeof(NthOutBatch) * threads * maxBeamSize
             + sizeof(float) * threads;
  //cerr << "shared=" << shared << endl;

  //HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
  //cerr << "step0" << endl;

  //BEGIN_TIMER("gBeamSizeInit");
  gBeamSizeInit<<<1, 1, 0, CudaStreamHandler::GetStream()>>>
    (hypo2BeamSizeWrap,
    batch2HypoWrap,
    hypo2CandidateWrap,
    isFirst,
    beamSizeSum,
    beamSizesWrap
    );
  //PAUSE_TIMER("gBeamSizeInit");

  /*
  cerr << "hypo2BeamSize=" << Debug(hypo2BeamSize, 2) << endl;
  cerr << "hypo2Candidate=" << Debug(hypo2Candidate, 2) << endl;
  cerr << "batch2Hypo=" << Debug(batch2Hypo, 2) << endl;
  cerr << endl;
  */
  //HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
  //cerr << "step1" << endl;

  //BEGIN_TIMER("gLogSoftMax");
  gLogSoftMax<<<blocks, threads, shared, CudaStreamHandler::GetStream()>>>
    (nBestCandidatesWrap,
     inWrap,
     b4Wrap,
     maxBeamSize,
     forbidUNK,
     hypo2BeamSizeWrap,
     hypo2CandidateWrap);
  //PAUSE_TIMER("gLogSoftMax");

  //HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
  //cerr << "step2" << endl;

  threads = 1;

  //BEGIN_TIMER("gNBestPerBatch");
  gNBestPerBatch<<<blocks, threads, 0, CudaStreamHandler::GetStream()>>>
    (nBestWrap,
     nBestCandidatesWrap,
     inWrap,
     costsWrap,
     maxBeamSize,
     forbidUNK,
     isFirst,
     hypo2BeamSizeWrap,
     batch2HypoWrap,
     hypo2CandidateWrap);
  //PAUSE_TIMER("gNBestPerBatch");

  //HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
  //cerr << "step3" << endl;
  //cerr << "3costs=" << Debug(costs, 0) << endl;
}

void TestMemCpy()
{
  using namespace std;

  cerr << "Starting" << endl;

  size_t NUM = 10;
  vector<float> h_vec1(NUM);
  for (size_t i = 0; i < NUM; ++i) {
    h_vec1[i] = i * 3;
  }

  TestMemCpy(NUM, h_vec1.data());

  cerr << "Finished" << endl;
}

}  // namespace mblas
}  // namespace GPU
}  // namespace amunmt
