#include "gpu/mblas/tensor_functions.h"
#include "gpu/mblas/handles.h"

using namespace std;

namespace amunmt {
namespace GPU {
namespace mblas {

thread_local CudaStreamHandler CudaStreamHandler::instance_;
thread_local CublasHandler CublasHandler::instance_;


Tensor& Swap(Tensor& Out, Tensor& In) {
  Out.swap(In);
  return Out;
}

__global__ void gMean(TensorWrapper<float> out,
                      const TensorWrapper<float> in,
                      const VectorWrapper<unsigned> sentenceLengths)
{
  // out = batches * states
  // in = max sentence length * states * 1 * batches
  // mapping = max length * batches

  int id = threadIdx.x + blockIdx.x * blockDim.x;
  //printf("id = %d in = %lu %lu %lu %lu = %lu %lu \n", id, in.dim(0), in.dim(1), in.dim(2), in.dim(3), in.size(), sizeof(in));

  if (id < out.size()) {
    unsigned indices[SHAPE_SIZE];
    out.id2Indices(id, indices);
    //printf("%d -> %lu %lu %lu %lu \n", id, indices[0], indices[1], indices[2], indices[3]);

    unsigned batch = indices[0];
    unsigned state = indices[1];

    float sum = 0.0f;
    int counter = 0;
    for (unsigned row = 0; row < in.dim(0); ++row) {
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

void Mean(Tensor& Out,
          const Tensor& In,
          const mblas::Vector<unsigned> &sentenceLengths)
{
  assert(Out.dim(2) == 1);
  assert(Out.dim(3) == 1);
  assert(Out.dim(0) == In.dim(3));
  assert(Out.dim(1) == In.dim(1));

  // mean of each ROW
  unsigned batchNum = Out.dim(0) * Out.dim(2) * Out.dim(3);
  unsigned stateLength = Out.dim(1);
  unsigned sentenceLength = (In.dim(0) * In.dim(2) * In.dim(3)) / batchNum;

  TensorWrapper<float> outWrap(Out);
  TensorWrapper<float> inWrap(In);
  //cerr << "outWrap=" << outWrap.Debug() << endl;

  VectorWrapper<unsigned> sentenceLengthsWrap(sentenceLengths);

  unsigned size = outWrap.size();
  unsigned threads = std::min((unsigned)MAX_THREADS, size);
  unsigned blocks =  (size / threads) + ((size % threads == 0) ?  0 : 1);

  gMean<<<blocks, threads, 0, CudaStreamHandler::GetStream()>>>
    (outWrap, inWrap, sentenceLengthsWrap);
  HANDLE_ERROR(cudaGetLastError());

}

__global__ void gWeightedMean(TensorWrapper<float> out,
                              const TensorWrapper<float> weights,
                              const TensorWrapper<float> in,
                              const VectorWrapper<unsigned> mapping
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
    for (unsigned i = 0; i < srcLen; ++i) {
      sum += weights(hypoInd, i) * in(i, stateInd, 0, batchInd);
    }

    out[id] = sum;
  }
}

void WeightedMean(Tensor& Out,const Tensor& Weights, const Tensor& In, const mblas::Vector<unsigned>& mapping)
{
  int numHypos = Weights.dim(0);
  int states = In.dim(1);

  Out.NewSize(numHypos, states);

  TensorWrapper<float> outWrap(Out);
  TensorWrapper<float> weightsWrap(Weights);
  TensorWrapper<float> inWrap(In);
  VectorWrapper<unsigned> mappingWrap(mapping);

  unsigned size = Out.size();
  unsigned nThreads = std::min((unsigned) MAX_THREADS, (unsigned)size);
  unsigned nBlocks =  (size / nThreads) + ((size % nThreads == 0) ?  0 : 1);

  gWeightedMean<<<nBlocks, nThreads, 0, CudaStreamHandler::GetStream()>>>
    (outWrap, weightsWrap, inWrap, mappingWrap);
  HANDLE_ERROR(cudaGetLastError());
  /*
  cerr << "nBlocks=" << nBlocks << endl;

  cerr << "Out=" << outWrap.Debug() << endl;
  cerr << "Weights=" << weightsWrap.Debug() << endl;
  cerr << "In=" << inWrap.Debug() << endl;
  cerr << "mapping=" << mapping.size() << endl;
  for (unsigned i = 0; i < mapping.size(); ++i) {
    cerr << mapping[i] << " ";
  }
  cerr << endl << endl;
  */
}

Tensor& Transpose(Tensor& Out, const Tensor& In) {
  unsigned m = In.dim(0);
  unsigned n = In.dim(1);

  Out.NewSize(n, m);

  float alpha = 1.0;
  float beta  = 0.0;

  HANDLE_ERROR_CUBLAS(cublasSgeam(CublasHandler::GetHandle(), CUBLAS_OP_T, CUBLAS_OP_T, m, n, &alpha, In.data(), n,
				&beta, In.data(), n, Out.data(), m));

  return Out;
}

Tensor& Transpose(Tensor& Out) {
  thread_local Tensor Temp;
  Transpose(Temp, Out);
  Swap(Out, Temp);
  return Out;
}

Tensor& Concat(Tensor& Out, const Tensor& In) {
  unsigned oldSize = Out.size();
  Out.Resize(Out.dim(0) + In.dim(0), Out.dim(1));

  mblas::copy(In.data(), In.size(), Out.data() + oldSize, cudaMemcpyDeviceToDevice);

  return Out;
}

Tensor& Copy(Tensor& Out, const Tensor& In) {
  Out.NewSize(In.dim(0), In.dim(1), In.dim(2), In.dim(3));

  mblas::copy(In.data(), In.size(), Out.data(), cudaMemcpyDeviceToDevice);

  return Out;
}

__global__ void gPasteRows(TensorWrapper<float> out,
                          const TensorWrapper<float> in,
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
    out(rowNo, inCol + colNo, 0, inRow) = in(inRow, inCol);
  }
}

void PasteRows(Tensor& Out, const Tensor& In, const unsigned rowNo, unsigned colNo)
{
  TensorWrapper<float> outWrap(Out);
  TensorWrapper<float> inWrap(In);

  unsigned size = In.size();
  unsigned nThreads = std::min((unsigned) MAX_THREADS, (unsigned)size);
  unsigned nBlocks =  (size / nThreads) + ((size % nThreads == 0) ?  0 : 1);

  gPasteRows<<<nBlocks, nThreads, 0, CudaStreamHandler::GetStream()>>>
    (outWrap, inWrap, rowNo, colNo);
  HANDLE_ERROR(cudaGetLastError());

}

Tensor& PasteRow(Tensor& Out,
                 const Tensor& In,
                 const unsigned r, const unsigned c)
{
  unsigned start = r * Out.dim(1) + c;

  mblas::copy(In.data(), In.size(), Out.data() + start, cudaMemcpyDeviceToDevice);

  return Out;
}

Tensor& CopyRow(Tensor& Out,
                const Tensor& In,
                const unsigned r, const unsigned c) {
  unsigned length = In.dim(1) - c;
  Out.NewSize(1, length);
  unsigned start = r * In.dim(1) + c;
  //unsigned end   = start + length;

  //mblas::copy(In.begin() + start, In.begin() + end, Out.begin());
  mblas::copy(In.data() + start, length , Out.data(), cudaMemcpyDeviceToDevice);

  return Out;
}

__global__ void gCopyRows(TensorWrapper<float> out,
                          const TensorWrapper<float> in,
                          const VectorWrapper<unsigned> indicesWrap)
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;

  if (id < out.size()) {
	  unsigned dim[SHAPE_SIZE];
	  out.id2Indices(id, dim);

	  unsigned indicesInd = dim[0];
	  unsigned inRow =indicesWrap[indicesInd];

      out(indicesInd, dim[1]) = in(inRow, dim[1]);

  }
}

Tensor& CopyRows(Tensor& Out,
                 const Tensor& In,
                 const mblas::Vector<unsigned>& indices)
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

  unsigned size = Out.size();

  unsigned numPairs = indices.size();

  TensorWrapper<float> outWrap(Out);
  const TensorWrapper<float> inWrap(In);
  const VectorWrapper<unsigned> indicesWrap(indices);
  //cerr << "size=" << size << endl;

  unsigned threads = std::min((unsigned) MAX_THREADS, (unsigned)size);
  unsigned blocks = size / threads + ((size % threads == 0) ?  0 : 1);

  gCopyRows<<<blocks, threads, 0, CudaStreamHandler::GetStream()>>>
    (outWrap, inWrap, indicesWrap);
  HANDLE_ERROR(cudaGetLastError());

  return Out;
}


Tensor& Assemble(Tensor& Out,
                 const Tensor& In,
                 const mblas::Vector<unsigned>& indices) {
  Out.NewSize(indices.size(), In.dim(1));
  //cerr << "Assemble=" << Out.Debug() << " " << In.Debug() << indices.size() << endl;

  CopyRows(Out, In, indices);
  return Out;
}

__global__ void gSlice(TensorWrapper<float> out,
                      const TensorWrapper<float> in,
                       unsigned n, unsigned dim)
{
  unsigned row = blockIdx.x;

  unsigned inCol = threadIdx.x + dim * n;
  unsigned outCol = threadIdx.x;

  while (outCol < out.dim(1)) {
    out(row, outCol) = in(row, inCol);

    inCol += blockDim.x;
    outCol += blockDim.x;
  }

}

Tensor& Slice(Tensor& Out,
              const Tensor& In,
              unsigned n, unsigned dim)
{
  assert(In.dim(2) == 1);
  assert(In.dim(3) == 1);

  Out.NewSize(In.dim(0), dim);

  TensorWrapper<float> outWrap(Out);
  const TensorWrapper<float> inWrap(In);

  /*
  cerr << "outWrap=" << outWrap.Debug() << endl;
  cerr << "inWrap=" << inWrap.Debug() << endl;
  cerr << "n=" << n << endl;
  cerr << "dim=" << dim << endl;
  cerr << endl;
  */

  unsigned threads = std::min((unsigned)MAX_THREADS, (unsigned)dim);
  unsigned blocks = In.dim(0);

  gSlice<<<blocks, threads, 0, CudaStreamHandler::GetStream()>>>
    (outWrap, inWrap, n, dim);
  HANDLE_ERROR(cudaGetLastError());

  return Out;
}

Tensor& Prod(cublasHandle_t handle, Tensor& C, const Tensor& A, const Tensor& B, bool transB)
{
  BEGIN_TIMER("Prod");
  assert((A.dim(2) == A.dim(3) == 1) || (B.dim(2) == B.dim(3) == 1));

  Tensor::value_type alpha = 1.0;
  Tensor::value_type beta = 0.0;

  unsigned m = A.dim(0) * A.dim(2) * A.dim(3);
  unsigned k = A.dim(1);
  unsigned mOut = A.dim(0);
  unsigned kOut = A.dim(1);

  /*
  if(transA) {
    std::swap(m, k);
    std::swap(mOut, kOut);
  }
  */
  unsigned l = B.dim(0) * B.dim(2) * B.dim(3);
  unsigned n = B.dim(1);
  unsigned lOut = B.dim(0);
  unsigned nOut = B.dim(1);
  if(transB) {
    std::swap(l, n);
    std::swap(lOut, nOut);
    }

  assert(k == l);

  unsigned lda = A.dim(1);
  unsigned ldb = B.dim(1);
  unsigned ldc = transB ? B.dim(0) * B.dim(2) * B.dim(3) : B.dim(1);

  unsigned dim2 = A.dim(2);
  if (transB) {
    // for GetAlignedSourceContext()
    assert((A.dim(2) == A.dim(3) == 1));
    C.NewSize(B.dim(2), nOut, 1, 1);
  }
  else {
    C.NewSize(mOut, nOut, A.dim(2) * B.dim(2), A.dim(3) * B.dim(3));
  }
  /*
  cerr << "C=" << C.Debug(0) << endl;
  cerr << "A=" << A.Debug(0) << endl;
  cerr << "B=" << B.Debug(0) << endl;
  cerr << "transB=" << transB << endl;
  cerr << m << " " << n << " " << k << endl;
  cerr << lda << " " << ldb << " " << ldc << endl;
  cerr << endl;
  */
  bool transA = false;
  cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

  HANDLE_ERROR_CUBLAS(cublasSgemm(handle, opB, opA,
                      n, m, k,
                      &alpha,
                      B.data(), ldb,
                      A.data(), lda,
                      &beta,
                      C.data(), ldc));
  PAUSE_TIMER("Prod");
  return C;
}

Tensor& Prod(Tensor& C, const Tensor& A, const Tensor& B,
             bool transB) {

  //std::cerr << "1C=" << C.Debug() << std::endl;
  //std::cerr << "1A=" << A.Debug() << std::endl;
  //std::cerr << "1B=" << B.Debug() << std::endl;

  Tensor &ret = Prod(CublasHandler::GetHandle(), C, A, B, transB);

  //std::cerr << "2C=" << C.Debug() << std::endl;
  return ret;
}

__global__ void gSoftMax(TensorWrapper<float> out,
                         const VectorWrapper<unsigned> batchIdsWrap,
                         const VectorWrapper<unsigned> sentenceLengthsWrap)
{
  extern __shared__ float _share[];

  unsigned numHypos = out.dim(0);
  unsigned maxLength = out.dim(1);

  int hypoInd =  blockIdx.x;
  int origSrcPos = threadIdx.x;

  while (hypoInd < numHypos) {
    unsigned batch = batchIdsWrap[hypoInd];
    unsigned length = sentenceLengthsWrap[batch];

    VectorWrapper<float> _max(_share, blockDim.x);

    if (origSrcPos < length) {
      _max[origSrcPos] = out(hypoInd, origSrcPos);
    }
    else {
      _max[origSrcPos] = LOWEST_FLOAT;
    }

    for (int tid = 0; tid < length; tid += blockDim.x) {
      int srcPos = tid + origSrcPos;
      if (srcPos < length) {
        float value = out(hypoInd, srcPos);

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
    VectorWrapper<float> _sum(_share, blockDim.x);

    _sum[origSrcPos] = 0.0f;
    for (int tid = 0; tid < maxLength; tid += blockDim.x) {
      int srcPos = tid + origSrcPos;
      if (srcPos < maxLength) {
        out(hypoInd, srcPos) = __expf(out(hypoInd, srcPos) - max);

        out(hypoInd, srcPos) *= srcPos < sentenceLengthsWrap[batch] ? 1 : 0; // sentencesMappingWrap(srcPos, batch, 0, 0);
        _sum[origSrcPos] += out(hypoInd, srcPos);
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
        out(hypoInd, srcPos) /= _sum[0];
      }
    }
    __syncthreads();
    hypoInd += gridDim.x;
  }
}

Tensor& Softmax(Tensor& Out,
                const mblas::Vector<unsigned>& batchIds,
                const mblas::Vector<unsigned> &sentenceLengths)
{
  unsigned numHypos = Out.dim(0);
  unsigned maxLength = Out.dim(1);

  TensorWrapper<float> outWrap(Out);
  const VectorWrapper<unsigned> batchIdsWrap(batchIds);
  const VectorWrapper<unsigned> sentenceLengthsWrap(sentenceLengths);

  int blocks = std::min(MAX_BLOCKS, (int)numHypos);
  int threads = std::min(MAX_THREADS, (int)maxLength);
  int shared = sizeof(float) * threads;

  /*
  std::cerr << "Out=" << Out.Debug(2) << std::endl;
  std::cerr << "batchIds=" << batchIds.Debug(1) << std::endl;
  std::cerr << "sentenceLengths=" << sentenceLengths.Debug(1) << std::endl;
  std::cerr << "blocks=" << blocks << std::endl;
  std::cerr << "threads=" << threads << std::endl;
  */

  gSoftMax<<<blocks, threads, shared, CudaStreamHandler::GetStream()>>>
    (outWrap, batchIdsWrap, sentenceLengthsWrap);
  HANDLE_ERROR(cudaGetLastError());

  /*
  std::cerr << "Out=" << Out.Debug(2) << std::endl;
  std::cerr << std::endl;
  */

  return Out;
}

__global__ void gLogSoftMax(TensorWrapper<float> out, unsigned shareSize)
{
  extern __shared__ float _share[];

  unsigned rows = out.dim(0);
  unsigned cols = out.dim(1);

  int rowIdx =  blockIdx.x;

  while (rowIdx < rows) {
    //float* _max = _share;
    VectorWrapper<float> _max(_share, shareSize);

    _max[threadIdx.x] = out(rowIdx, threadIdx.x);
    for (int tid = 0; tid < cols; tid += blockDim.x) {
      int id = tid + threadIdx.x;
      if (id < cols) {
        const float &val = out(rowIdx, id);
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
    VectorWrapper<float> _sum(_share, shareSize);

    _sum[threadIdx.x] = 0.0f;
    for (int tid = 0; tid < cols; tid += blockDim.x) {
      int id = tid + threadIdx.x;
      if (id < cols) {
        //row[id] = exp(row[id] - max);
        float &val = out(rowIdx, id);
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
        float &val = out(rowIdx, id);
        val = __logf(val /_sum[0]);
      }
    }
    __syncthreads();
    rowIdx += gridDim.x;
  }
}


Tensor& LogSoftmax(Tensor& Out)
{
  TensorWrapper<float> outWrap(Out);

  int blocks = std::min(MAX_BLOCKS, (int)Out.dim(0));
  int threads = std::min(MAX_THREADS, (int)Out.dim(1));
  int shared = sizeof(float) * threads;

  gLogSoftMax<<<blocks, threads, shared, CudaStreamHandler::GetStream()>>>
    (Out, threads);
  HANDLE_ERROR(cudaGetLastError());

  return Out;
}

__global__ void gSetColumn(TensorWrapper<float> in, int noColumn, float value) {
  int n_rows = in.dim(0);

  int rowNumber = threadIdx.x  + blockDim.x * blockIdx.x;

  if (rowNumber < n_rows) {
    in(rowNumber, noColumn) = value;
  }
}

void SetColumn(Tensor& In, int noColumn, float value) {
  int nRows = In.dim(0);
  int nBlocks = nRows / MAX_THREADS + ((nRows % MAX_THREADS == 0) ?  0 : 1);
  int nThreads = std::min(MAX_THREADS, nRows);

  TensorWrapper<float> inWrap(In);

  gSetColumn<<<nBlocks, nThreads, 0, mblas::CudaStreamHandler::GetStream()>>>
    (inWrap, noColumn, value);
  HANDLE_ERROR(cudaGetLastError());
}

__global__ void gFill(TensorWrapper<float> in, float val) {
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  if (index < in.size()) {
    in[index] = val;
  }
}

void Fill(Tensor& In, float value) {
  unsigned size = In.size();

  if (value) {
    int nThreads = std::min(MAX_THREADS, (int)size);
    int nBlocks = (size / nThreads) + ((size % nThreads == 0) ? 0 : 1);

    TensorWrapper<float> inWrap(In);

    gFill<<<nBlocks, nThreads, 0, CudaStreamHandler::GetStream()>>>
      (inWrap, value);
    HANDLE_ERROR(cudaGetLastError());
  }
  else {
    HANDLE_ERROR(cudaMemsetAsync(In.data(), 0, size * sizeof(float), CudaStreamHandler::GetStream()));
  }

}

__global__
void gMapMatrix(TensorWrapper<float> in,
                const VectorWrapper<unsigned> sentenceLengthsWrap,
                int i)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < in.size()) {
    int numCols = in.dim(1);
    int batchIdx = tid / numCols;
    int col = tid % numCols;

    //in[tid] *= mappingWrap(i, batchIdx, 0, 0);
    in(batchIdx, col) *= (i < sentenceLengthsWrap[batchIdx] ? 1 : 0);
  }
}

void MapMatrix(Tensor& state,
              const mblas::Vector<unsigned> &sentenceLengths,
              unsigned i)
{
  // blank out rows in the state matrix where the word position i does not exist
  // mapping is a concatenated array of 1 & 0 of each sentence in the batch to say whether word exists or not.

  int batchSize = state.dim(0);
  int stateLength = state.dim(1);

  int numThreads = std::min((int)state.size(), MAX_THREADS);
  int numBlocks = (state.size() / numThreads) + ((state.size() % numThreads == 0) ? 0 : 1);

  TensorWrapper<float> stateWrap(state);
  VectorWrapper<unsigned> sentenceLengthsWrap(sentenceLengths);

  gMapMatrix<<<numBlocks, numThreads, 0, CudaStreamHandler::GetStream()>>>
    (stateWrap, sentenceLengthsWrap, i);
  HANDLE_ERROR(cudaGetLastError());

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

__device__ unsigned getIndex(const dim3 &dim, const dim3 &val)
{
  unsigned ret = dim.x * val.x + dim.y * val.y + dim.z * val.z;
  return ret;
}


__global__ void gLNormalization(TensorWrapper<float> out,
                                const TensorWrapper<float> in,
                                const TensorWrapper<float> alphaWrap,
                                const TensorWrapper<float> betaWrap,
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

void Normalization(Tensor &out,
                  const Tensor &in,
                  const Tensor &alpha,
                  const Tensor *beta,
                  float eps)
{
  assert(in.dim(0) < MAX_BLOCKS);
  assert(in.dim(2) < MAX_BLOCKS);
  assert(in.dim(3) < MAX_BLOCKS);

  //out.Reshape(in.dim(0), in.dim(1), in.dim(2), in.dim(3));

  int numThreads = std::min((unsigned) in.dim(1), (unsigned) MAX_THREADS);
  dim3 numBlocks(in.dim(0), in.dim(2), in.dim(3));
  int shared = numThreads * sizeof(float) * 2;

  TensorWrapper<float> outWrap(out);
  const TensorWrapper<float> inWrap(in);
  const TensorWrapper<float> alphaWrap(alpha);
  TensorWrapper<float> *betaWrap = beta ? new TensorWrapper<float>(*beta) : new TensorWrapper<float>();

  gLNormalization<<<numBlocks, numThreads, shared, CudaStreamHandler::GetStream()>>>
    (outWrap, inWrap, alphaWrap, *betaWrap, eps);
  HANDLE_ERROR(cudaGetLastError());

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

void Normalization(Tensor& out, const Tensor& in, const Tensor& alpha, const Tensor& beta,
                       float eps)
{
  Normalization(out, in, alpha, &beta, eps);
}

void Normalization(Tensor& out, const Tensor& in, const Tensor& alpha, float eps)
{
  Normalization(out, in, alpha, nullptr, eps);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////

__global__
void gBeamSizeInit(VectorWrapper<unsigned> hypo2BeamSizeWrap,
                    VectorWrapper<unsigned> batch2HypoWrap,
                    VectorWrapper<unsigned> hypo2CandidateWrap,
                    bool isFirst,
                    unsigned beamSizeSum,
                    const VectorWrapper<unsigned> beamSizesWrap)
{
  unsigned hypoInd = 0;
  unsigned candidateInd = 0;

  unsigned a = 0, b = 0;
  //printf("beamSizesWrap.size()=%u \n", beamSizesWrap.size());
  for (unsigned batchInd = 0; batchInd < beamSizesWrap.size(); ++batchInd) {
    unsigned beamSize = beamSizesWrap[batchInd];
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
        for (unsigned j = 0; j < beamSize; ++j) {
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
float GetMaxScore(const TensorWrapper<NthOutBatch> &nBestMatrix)
{
  float ret = LOWEST_FLOAT;
  for (unsigned i = 0; i < nBestMatrix.dim(1); ++i) {
      const NthOutBatch &curr = nBestMatrix[i];
      if (curr.score > ret) {
        ret = curr.score;
    }
  }

  return ret;
}

__device__
void AddElement(float &minScore,
    unsigned &i,
    VectorWrapper<NthOutBatch> &vec,
    bool forbidUNK,
    unsigned vocabInd,
    const NthOutBatch &ele)
{
  const float score = ele.score;

  if (forbidUNK && vocabInd == UNK_ID) {
    vec[i].score = LOWEST_FLOAT;
    minScore = LOWEST_FLOAT;
  }
  else {
    vec[i] = ele;

    if (score < minScore) {
      minScore = score;
    }

    ++i;
  }

}

__device__
void MergeElement(float &minScore,
                  VectorWrapper<NthOutBatch> &vec,
                  unsigned arrSize,
                  const NthOutBatch &ele)
{
  float newMinScore = HIGHEST_FLOAT;
  bool found = false;
  for (unsigned i = 0; i < arrSize; ++i) {
    NthOutBatch &currEle = vec[i];
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
                  VectorWrapper<NthOutBatch> &vec,
                  unsigned arrSize,
                  const NthOutBatch &ele,
                  bool forbidUNK,
                  unsigned vocabInd)
{
  if (forbidUNK && vocabInd == UNK_ID) {
    // do nothing
  }
  else if (ele.score > minScore) {
    // replace element with min score
    MergeElement(minScore, vec, arrSize, ele);

    /*
    printf("arrInd=%d ind=%d vocabId=%d \n",
          arrInd,
          _max[threadIdx.x].ind,
          _max[threadIdx.x].vocabId);
    */
  }
}

__device__
void MaxAndSum(float &max, float &tot, const float &val)
{
  if (val > max) {
    float delta = max - val; // val - max; // TODO see LogSoftmaxFn
    tot *= __expf(delta);

    max = val;
    tot += 1; // exp(val - max) = exp(0) = 1
  }
  else {
    tot += __expf(val - max);
  }
}

__device__
void NBestAndMaxAndSum(VectorWrapper<NthOutBatch> &nBestCandidatesWrap,
                TensorWrapper<NthOutBatch> &nBestMatrix,
                VectorWrapper<float> &max,
                VectorWrapper<float> &sum,
                const TensorWrapper<float> &in,
                const TensorWrapper<float> &b4Wrap,
                const unsigned hypoInd,
                const unsigned maxBeamSize,
                const bool forbidUNK,
                const VectorWrapper<unsigned> &hypo2BeamSizeWrap,
                const VectorWrapper<unsigned> &hypo2CandidateWrap,
                bool requireProb)
{
  assert(max.size() == blockDim.x);
  assert(sum.size() == blockDim.x);

  VectorWrapper<NthOutBatch> row = nBestMatrix.Row(threadIdx.x);

  unsigned vocabSize = in.dim(1);

  assert(hypoInd < hypo2BeamSizeWrap.size());
  unsigned beamSize = hypo2BeamSizeWrap[hypoInd];

  float minScore = HIGHEST_FLOAT;

  // init
  unsigned vocabInd = threadIdx.x;

  max[threadIdx.x] = LOWEST_FLOAT;
  sum[threadIdx.x] = 0.0f;

  unsigned i = 0;
  while (vocabInd < vocabSize && i < beamSize) {
    const float score = in(hypoInd, vocabInd) + b4Wrap(0, vocabInd);

    unsigned arrInd = hypoInd * vocabSize + vocabInd;
    NthOutBatch ele(arrInd, score, hypoInd, vocabInd);

    AddElement(minScore, i, row, forbidUNK, vocabInd, ele);

    // max & sum
    if (requireProb) {
      MaxAndSum(max[threadIdx.x], sum[threadIdx.x], score);
    }

    vocabInd += blockDim.x;
  }

  // MAIN LOOP
  while (vocabInd < vocabSize) {
    const float score = in(hypoInd, vocabInd) + b4Wrap(0, vocabInd);
    unsigned arrInd = hypoInd * vocabSize + vocabInd;
    NthOutBatch ele(arrInd, score, hypoInd, vocabInd);

    MergeElement(minScore, row, beamSize, ele, forbidUNK, vocabInd);

    // max & sum
    MaxAndSum(max[threadIdx.x], sum[threadIdx.x], score);

    vocabInd += blockDim.x;
  } // while (vocabInd < vocabSize) {

  // merge nbest from different threads
  int len = blockDim.x;
  while (len != 1) {
    __syncthreads();
    int skip = (len + 1) >> 1;
    if (threadIdx.x < (len >> 1)) {

      for (unsigned i = 0; i < beamSize; ++i) {
        const NthOutBatch &ele = nBestMatrix(threadIdx.x + skip, i);
        if (ele.score > minScore) {
          MergeElement(minScore, row, beamSize, ele);
        }
      }
    }
    len = (len + 1) >> 1;

  }

  __syncthreads();
  if (threadIdx.x == 0) {
    // copy to output array
    assert(hypoInd < hypo2CandidateWrap.size());
    unsigned candidateInd = hypo2CandidateWrap[hypoInd];
    for (unsigned i = 0; i < beamSize; ++i) {
      const NthOutBatch &curr = nBestMatrix(0, i);
      //printf("vocabInd=%u \n", best.vocabInd);

      assert(candidateInd + i < nBestCandidatesWrap.size());
      nBestCandidatesWrap[candidateInd + i] = curr;
    }
  }

  // top score and sum
  if (requireProb) {
    unsigned size = max.size();
    unsigned len = (size + 1) >> 1;
    //printf("size=%i %i \n", size, len);

    unsigned ind = threadIdx.x;
    float &max0 = max[ind];
    float &sum0 = sum[ind];

    while (len) {
      __syncthreads();
      //printf("size=%i %i \n", size, len);

      unsigned otherInd = ind + len;

      if (otherInd < size) {

        const float &maxOther = max[otherInd];
        const float &sumOther = sum[otherInd];

        if (max0 > maxOther) {
          float delta = maxOther - max0;
          sum0 = sum0 + __expf(delta) * sumOther;
        }
        else {
          float delta = max0 - maxOther;
          sum0 = __expf(delta) * sum0 + sumOther;

          max0 = maxOther;
        }

      }

      size = len;
      len = (len > 1) ? (len + 1) >> 1 : 0;
    }
  }


}

///////////////////////////////////////////////////////////////////////////////////////////////////////
__device__
void dLogSoftMax(VectorWrapper<NthOutBatch> &nBestCandidatesWrap,
                      const TensorWrapper<float> &in,
                      const TensorWrapper<float> &b4Wrap,
                      const unsigned hypoInd,
                      const unsigned maxBeamSize,
                      const float topScore,
                      const float sumExp,
                      const VectorWrapper<unsigned> &hypo2BeamSizeWrap,
                      const VectorWrapper<unsigned> &hypo2CandidateWrap)
{
  unsigned vocabSize = in.dim(1);

  // apply partition and log to top
  if (threadIdx.x == 0) {
    //printf("sum=%f \n", sum[0]);
    //printf("val=%f %f \n", in(rowIdx, ele.vocabId, 0, 0), val);

    // nbest
    unsigned beamSize = hypo2BeamSizeWrap[hypoInd];
    unsigned startPos = hypo2CandidateWrap[hypoInd];
    for (unsigned i = 0; i < beamSize; ++i) {
      NthOutBatch &ele = nBestCandidatesWrap[startPos + i];

      float &val = ele.score;
      val = __expf(val - topScore);
      val = __logf(val /sumExp);
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void gLogSoftMax(VectorWrapper<NthOutBatch> nBestCandidatesWrap,
                        const TensorWrapper<float> in,
                        const TensorWrapper<float> b4Wrap,
                        unsigned maxBeamSize,
                        bool forbidUNK,
                        const VectorWrapper<unsigned> hypo2BeamSizeWrap,
                        const VectorWrapper<unsigned> hypo2CandidateWrap,
                        bool requireProb)
{
  extern __shared__ char _sharePtr[];

  void *ptrOffset = _sharePtr;
  VectorWrapper<float> max((float*)ptrOffset, blockDim.x);

  ptrOffset = _sharePtr + sizeof(float) * blockDim.x;
  VectorWrapper<float> sum((float*)ptrOffset, blockDim.x);

  ptrOffset = _sharePtr + 2 * sizeof(float) * blockDim.x;
  TensorWrapper<NthOutBatch> nBestMatrix((NthOutBatch*)ptrOffset, blockDim.x, maxBeamSize, 1, 1);

  unsigned hypos = in.dim(0);
  unsigned vocabSize = in.dim(1);

  unsigned hypoInd =  blockIdx.x; // index of previous hypo
  while (hypoInd < hypos) {
    NBestAndMaxAndSum(nBestCandidatesWrap,
                nBestMatrix,
                max,
                sum,
                in,
                b4Wrap,
                hypoInd,
                maxBeamSize,
                forbidUNK,
                hypo2BeamSizeWrap,
                hypo2CandidateWrap,
                requireProb);
    __syncthreads();

    if (requireProb) {
      const float topScore = max[0];
      const float sumExp = sum[0];

      dLogSoftMax(nBestCandidatesWrap,
                      in,
                      b4Wrap,
                      hypoInd,
                      maxBeamSize,
                      topScore,
                      sumExp,
                      hypo2BeamSizeWrap,
                      hypo2CandidateWrap);
    }

    __syncthreads();
    hypoInd += gridDim.x;
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void gNBestPerBatch(VectorWrapper<NthOutBatch> nBestWrap,
                        VectorWrapper<NthOutBatch> nBestCandidatesWrap,
                        const TensorWrapper<float> in,
                        const VectorWrapper<float> costsWrap,
                        unsigned maxBeamSize,
                        bool forbidUNK,
                        bool isFirst,
                        const VectorWrapper<unsigned> hypo2BeamSizeWrap,
                        const VectorWrapper<unsigned> batch2HypoWrap,
                        const VectorWrapper<unsigned> hypo2CandidateWrap)
{
  //unsigned rows = in.dim(0);
  unsigned batchSize = batch2HypoWrap.size();

  unsigned batchInd =  blockIdx.x;
  while (batchInd < batchSize) {
    assert(batchInd < batch2HypoWrap.size());
    assert(batchInd < hypo2BeamSizeWrap.size());
    assert(batchInd < nBestWrap.size());

    unsigned hypoInd = batch2HypoWrap[batchInd];
    unsigned beamSize = hypo2BeamSizeWrap[hypoInd];
    assert(beamSize);

    unsigned nextHypoInd;
    if (isFirst) {
      nextHypoInd = batchInd * beamSize;
    }
    else {
      nextHypoInd = hypoInd;
    }

    // candiate from 1st hypo
    float minScore = HIGHEST_FLOAT;
    assert(hypoInd < hypo2CandidateWrap.size());
    unsigned candidateInd = hypo2CandidateWrap[hypoInd];
    for (unsigned i = 0; i < beamSize; ++i) {
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
      assert(nextHypoInd < nBestWrap.size());
      VectorWrapper<NthOutBatch> offset = nBestWrap.Offset(nextHypoInd);

      for (unsigned hypoOffset = 1; hypoOffset < beamSize; ++hypoOffset) {
        //printf("hypoInd=%d \n", (hypoInd + hypoOffset));

        //printf("prevHypoInd=%, candidateInd=%d \n", prevHypoInd, candidateInd);
        assert((hypoInd + hypoOffset) < costsWrap.size());
        float prevCost = costsWrap[hypoInd + hypoOffset];

        assert((hypoInd + hypoOffset) < hypo2CandidateWrap.size());
        unsigned candidateInd = hypo2CandidateWrap[hypoInd + hypoOffset];

        for (unsigned candidateOffset = 0; candidateOffset < beamSize; ++candidateOffset) {
          assert((candidateInd + candidateOffset) < nBestCandidatesWrap.size());
          NthOutBatch &candidate = nBestCandidatesWrap[candidateInd + candidateOffset];
          candidate.score += prevCost;

          assert(nextHypoInd < nBestWrap.size());
          NthOutBatch *arr = &nBestWrap[nextHypoInd];

          if (candidate.score > minScore) {
            MergeElement(minScore, offset, beamSize, candidate);
          }
        }
      }
    }

    batchInd += gridDim.x;
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////

void LogSoftmaxAndNBest(mblas::Vector<NthOutBatch> &nBest,
                const Tensor& in,
                const Tensor& b4,
                const mblas::Vector<float> &costs,
                bool forbidUNK,
                unsigned maxBeamSize,
                const std::vector<unsigned>& beamSizes,
                unsigned beamSizeSum,
                bool isFirst,
                bool requireProb)
{
  //BEGIN_TIMER("LogSoftmax excl kernels");

  bool safe = (maxBeamSize * MAX_THREADS) < in.dim(1);
  if (!safe) {
    cerr << "The target vocab size looks too small for the fused softmax function. If you experience a crash, add '--use-fused-softmax false' when running amun" << endl;
  }

  //cerr << "in=" << in.Debug(0) << endl;
  //cerr << "beamSizes=" << beamSizes.size() << endl;

  // create beam size vectors on GPU but exclude empty beams
  unsigned batchSize = 0;
  unsigned candidateInd = 0;
  for (unsigned batchInd = 0; batchInd < beamSizes.size(); ++batchInd) {
    unsigned beamSize = beamSizes[batchInd];
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

  mblas::Vector<unsigned> d_beamSizes(beamSizes);
  mblas::Vector<unsigned> hypo2BeamSize(in.dim(0));
  mblas::Vector<unsigned> hypo2Candidate(in.dim(0));
  mblas::Vector<unsigned> batch2Hypo(batchSize);
  mblas::Vector<NthOutBatch> nBestCandidates(candidateInd);

  /*
  cerr << "in=" << in.Debug(1) << endl;
  cerr << "beamSizes=" << beamSizes.size() << endl;
  cerr << "beamSizeSum=" << beamSizeSum << endl;
  cerr << "batchSize=" << batchSize << endl;
  cerr << "candidateInd=" << candidateInd << endl;
  cerr << "hypo2BeamSize=" << hypo2BeamSize.Debug(0) << endl;
  cerr << "hypo2Candidate=" << hypo2Candidate.Debug(0) << endl;
  cerr << "batch2Hypo=" << batch2Hypo.Debug(0) << endl;
  cerr << "nBest=" << nBest.Debug(0) << endl;
  cerr << "nBestCandidates=" << nBestCandidates.Debug(0) << endl;
  cerr << endl;
  */

  TensorWrapper<float> inWrap(in);
  TensorWrapper<float> b4Wrap(b4);
  VectorWrapper<unsigned> hypo2BeamSizeWrap(hypo2BeamSize);
  VectorWrapper<unsigned> hypo2CandidateWrap(hypo2Candidate);
  VectorWrapper<unsigned> batch2HypoWrap(batch2Hypo);
  VectorWrapper<NthOutBatch> nBestWrap(nBest);
  VectorWrapper<NthOutBatch> nBestCandidatesWrap(nBestCandidates);
  VectorWrapper<float> costsWrap(costs);

  VectorWrapper<unsigned> beamSizesWrap(d_beamSizes);

  //PAUSE_TIMER("LogSoftmax excl kernels");
  //HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
      
  //BEGIN_TIMER("gBeamSizeInit");
  gBeamSizeInit<<<1, 1, 0, CudaStreamHandler::GetStream()>>>
    (hypo2BeamSizeWrap,
    batch2HypoWrap,
    hypo2CandidateWrap,
    isFirst,
    beamSizeSum,
    beamSizesWrap
    );
  HANDLE_ERROR(cudaGetLastError());
  //PAUSE_TIMER("gBeamSizeInit");
  
  /*
  cerr << "hypo2BeamSize=" << Debug(hypo2BeamSize, 2) << endl;
  cerr << "hypo2Candidate=" << Debug(hypo2Candidate, 2) << endl;
  cerr << "batch2Hypo=" << Debug(batch2Hypo, 2) << endl;
  cerr << endl;
  */
  //HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
    
  unsigned blocks = std::min((unsigned) MAX_BLOCKS, in.dim(0));
  unsigned threads = std::min((unsigned)MAX_THREADS, in.dim(1));
  unsigned shared = sizeof(NthOutBatch) * threads * maxBeamSize
             + 2 * sizeof(float) * threads;

  //BEGIN_TIMER("gLogSoftMax");
  gLogSoftMax<<<blocks, threads, shared, CudaStreamHandler::GetStream()>>>
    (nBestCandidatesWrap,
     inWrap,
     b4Wrap,
     maxBeamSize,
     forbidUNK,
     hypo2BeamSizeWrap,
     hypo2CandidateWrap,
     requireProb);
  HANDLE_ERROR(cudaGetLastError());
  //PAUSE_TIMER("gLogSoftMax");
  
  //HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
  blocks = std::min((unsigned) MAX_BLOCKS, batchSize);

  //BEGIN_TIMER("gNBestPerBatch");
  gNBestPerBatch<<<blocks, 1, 0, CudaStreamHandler::GetStream()>>>
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
  HANDLE_ERROR(cudaGetLastError());
  //PAUSE_TIMER("gNBestPerBatch");
  
  //HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
  //cerr << "3costs=" << Debug(costs, 0) << endl;
}

void TestMemCpy()
{
  using namespace std;

  cerr << "Starting" << endl;

  unsigned NUM = 10;
  vector<float> h_vec1(NUM);
  for (unsigned i = 0; i < NUM; ++i) {
    h_vec1[i] = i * 3;
  }

  TestMemCpy(NUM, h_vec1.data());

  cerr << "Finished" << endl;
}

}  // namespace mblas
}  // namespace GPU
}  // namespace amunmt
