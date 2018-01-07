#include "common/histories.h"
#include "common/utils.h"
#include "gpu/mblas/matrix_functions.h"
#include "gpu/mblas/handles.h"
#include "gpu/decoder/enc_out_gpu.h"
#include "gpu/dl4mt/cellstate.h"

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
                      const VectorWrapper<unsigned> sentenceLengths)
{
  // out = batches * states
  // in = max sentence length * states * 1 * batches
  // mapping = max length * batches

  unsigned id = threadIdx.x + blockIdx.x * blockDim.x;
  //printf("id = %d in = %lu %lu %lu %lu = %lu %lu \n", id, in.dim(0), in.dim(1), in.dim(2), in.dim(3), in.size(), sizeof(in));

  if (id < out.GetShape().size()) {
    unsigned indices[SHAPE_SIZE];
    out.GetShape().id2Indices(id, indices);
    //printf("%d -> %lu %lu %lu %lu \n", id, indices[0], indices[1], indices[2], indices[3]);

    unsigned batch = indices[0];
    unsigned state = indices[1];

    float sum = 0.0f;
    unsigned counter = 0;
    for (unsigned row = 0; row < in.GetShape().dim(0); ++row) {
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
  //cerr << "Out=" << Out.Debug() << endl;

  VectorWrapper<unsigned> sentenceLengthsWrap(sentenceLengths);

  unsigned size = Out.size();
  unsigned threads = std::min(MAX_THREADS, size);
  unsigned blocks =  (size / threads) + ((size % threads == 0) ?  0 : 1);

  gMean<<<blocks, threads, 0, CudaStreamHandler::GetStream()>>>
    (Out, In, sentenceLengthsWrap);

}

__global__ void gWeightedMean(MatrixWrapper<float> out,
                              const MatrixWrapper<float> weights,
                              const MatrixWrapper<float> in,
                              const VectorWrapper<unsigned> hypo2Batch
                              )
{
  unsigned numHypos = weights.GetShape().dim(0);
  unsigned states = in.GetShape().dim(1);
  unsigned srcLen = weights.GetShape().dim(1);

  unsigned id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id < numHypos * states) {
    unsigned hypoInd = id / states;
    unsigned batchInd = hypo2Batch[hypoInd];
    unsigned stateInd = id % states;
    //printf("hypoInd=%d batchInd=%d stateInd=%d \n", hypoInd, batchInd, stateInd);

    float sum = 0.0f;
    for (unsigned i = 0; i < srcLen; ++i) {
      sum += weights(hypoInd, i) * in(i, stateInd, 0, batchInd);
    }

    out[id] = sum;
  }
}

void WeightedMean(Matrix& out,const Matrix& weights, const Matrix& in, const mblas::Vector<unsigned>& hypo2Batch)
{
  unsigned numHypos = weights.dim(0);
  unsigned states = in.dim(1);

  out.NewSize(numHypos, states);

  unsigned size = out.size();
  unsigned nThreads = std::min(MAX_THREADS, size);
  unsigned nBlocks =  (size / nThreads) + ((size % nThreads == 0) ?  0 : 1);
  /*
  cerr << "nBlocks=" << nBlocks << endl;
  cerr << "Out=" << out.Debug(0) << endl;
  cerr << "Weights=" << weights.Debug(0) << endl;
  cerr << "In=" << in.Debug(0) << endl;
  cerr << "hypo2Batch=" << hypo2Batch.Debug(1) << endl;
  cerr << endl << endl;
  */
  gWeightedMean<<<nBlocks, nThreads, 0, CudaStreamHandler::GetStream()>>>
    (out, weights, in, hypo2Batch);
}

Matrix& Transpose(Matrix& Out, const Matrix& In) {
  unsigned m = In.dim(0);
  unsigned n = In.dim(1);

  Out.NewSize(n, m);

  float alpha = 1.0;
  float beta  = 0.0;

  HANDLE_ERROR_CUBLAS(cublasSgeam(CublasHandler::GetHandle(), CUBLAS_OP_T, CUBLAS_OP_T, m, n, &alpha, In.data(), n,
				&beta, In.data(), n, Out.data(), m));

  return Out;
}

Matrix& Transpose(Matrix& Out) {
  thread_local Matrix Temp;
  Transpose(Temp, Out);
  Swap(Out, Temp);
  return Out;
}

Matrix& Concat(Matrix& Out, const Matrix& In) {
  unsigned oldSize = Out.size();
  Out.Resize(Out.dim(0) + In.dim(0), Out.dim(1));

  mblas::copy(In.data(), In.size(), Out.data() + oldSize, cudaMemcpyDeviceToDevice);

  return Out;
}

Matrix& Copy(Matrix& Out, const Matrix& In) {
  Out.NewSize(In.dim(0), In.dim(1), In.dim(2), In.dim(3));

  mblas::copy(In.data(), In.size(), Out.data(), cudaMemcpyDeviceToDevice);

  return Out;
}

__global__ void gPasteRows(MatrixWrapper<float> out,
                          const MatrixWrapper<float> in,
                          unsigned rowNo, unsigned colNo)
{
  unsigned inRows = in.GetShape().dim(0);
  unsigned inCols = in.GetShape().dim(1);

  unsigned id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id < inRows * inCols) {
    unsigned outCols = out.GetShape().dim(1);

    unsigned inRow = id / inCols;
    unsigned inCol = id % inCols;

    //out[outID] = in[id];
    out(rowNo, inCol + colNo, 0, inRow) = in(inRow, inCol);
  }
}

void PasteRows(Matrix& Out, const Matrix& In, const unsigned rowNo, unsigned colNo)
{
  MatrixWrapper<float> outWrap(Out);
  MatrixWrapper<float> inWrap(In);

  unsigned size = In.size();
  unsigned nThreads = std::min(MAX_THREADS, size);
  unsigned nBlocks =  (size / nThreads) + ((size % nThreads == 0) ?  0 : 1);

  gPasteRows<<<nBlocks, nThreads, 0, CudaStreamHandler::GetStream()>>>
    (outWrap, inWrap, rowNo, colNo);

}

Matrix& PasteRow(Matrix& Out,
                 const Matrix& In,
                 const unsigned r, const unsigned c)
{
  unsigned start = r * Out.dim(1) + c;

  mblas::copy(In.data(), In.size(), Out.data() + start, cudaMemcpyDeviceToDevice);

  return Out;
}

Matrix& CopyRow(Matrix& Out,
                const Matrix& In,
                const unsigned r, const unsigned c) {
  unsigned length = In.dim(1) - c;
  Out.NewSize(1, length);
  unsigned start = r * In.dim(1) + c;
  //unsigned end   = start + length;

  //mblas::copy(In.begin() + start, In.begin() + end, Out.begin());
  mblas::copy(In.data() + start, length , Out.data(), cudaMemcpyDeviceToDevice);

  return Out;
}

__global__ void gCopyRows(MatrixWrapper<float> out,
                          const MatrixWrapper<float> in,
                          const VectorWrapper<unsigned> inRows)
{
  unsigned id = threadIdx.x + blockIdx.x * blockDim.x;

  if (id < out.GetShape().size()) {
	  unsigned dim[SHAPE_SIZE];
	  out.GetShape().id2Indices(id, dim);

	  unsigned indicesInd = dim[0];
	  unsigned inRow =inRows[indicesInd];

    out(indicesInd, dim[1]) = in(inRow, dim[1]);

  }
}

Matrix& CopyRows(Matrix& out,
                 const Matrix& in,
                 const mblas::Vector<unsigned>& inRows)
{
  /*
  cerr << "out=" << out.Debug(0) << endl;
  cerr << "in=" << in.Debug(0) << endl;
  cerr << "inRows=" << inRows.Debug(2) << endl;
  cerr << endl;
  */
  assert(in.dim(1) == out.dim(1));
  assert(out.dim(0) == inRows.size());

  assert(in.dim(2) == 1);
  assert(in.dim(3) == 1);
  assert(out.dim(2) == 1);
  assert(out.dim(3) == 1);

  unsigned size = out.size();
  unsigned numPairs = inRows.size();
  //cerr << "size=" << size << endl;

  unsigned threads = std::min(MAX_THREADS, size);
  unsigned blocks = size / threads + ((size % threads == 0) ?  0 : 1);

  gCopyRows<<<blocks, threads, 0, CudaStreamHandler::GetStream()>>>
    (out, in, inRows);

  return out;
}

__global__ void gCopyRows(MatrixWrapper<float> out,
                          const MatrixWrapper<float> in,
                          const VectorWrapper<unsigned> inRows,
                          const VectorWrapper<unsigned> outRows,
                          const Shape shape)
{
  unsigned id = threadIdx.x + blockIdx.x * blockDim.x;

  if (id < shape.size()) {
    unsigned dim[SHAPE_SIZE];
    shape.id2Indices(id, dim);

    unsigned indicesInd = dim[0];
    unsigned inRow  = inRows[indicesInd];
    unsigned outRow = outRows[indicesInd];

    out(outRow, dim[1]) = in(inRow, dim[1]);

  }
}

Matrix& CopyRows(Matrix& out,
                 const Matrix& in,
                 const mblas::Vector<unsigned>& inRows,
                 const mblas::Vector<unsigned>& outRows)
{
  /*
  cerr << "out=" << out.Debug(0) << endl;
  cerr << "in=" << in.Debug(0) << endl;
  cerr << "inRows=" << inRows.Debug(2) << endl;
  cerr << "outRows=" << outRows.Debug(0) << endl;
  cerr << endl;
  */
  assert(inRows.size() == outRows.size());
  //assert(in.dim(0) >= inRows.size());
  assert(out.dim(0) >= inRows.size());

  assert(in.dim(1) == out.dim(1));
  assert(in.dim(2) == 1);
  assert(in.dim(3) == 1);
  assert(out.dim(2) == 1);
  assert(out.dim(3) == 1);

  Shape shape(inRows.size(), out.dim(1), 1, 1);
  unsigned size = shape.size();
  //cerr << "size=" << size << endl;

  unsigned threads = std::min(MAX_THREADS, size);
  unsigned blocks = size / threads + ((size % threads == 0) ?  0 : 1);

  gCopyRows<<<blocks, threads, 0, CudaStreamHandler::GetStream()>>>
    (out, in, inRows, outRows, shape);

  return out;

}


Matrix& Assemble(Matrix& out,
                 const Matrix& in,
                 const mblas::Vector<unsigned>& inRows)
{
  out.NewSize(inRows.size(), in.dim(1));
  //cerr << "Assemble=" << out.Debug() << " " << in.Debug() << indices.size() << endl;

  CopyRows(out, in, inRows);
  return out;
}

Matrix& AssembleTopup(Matrix& out,
                 const Matrix& in,
                 const mblas::Vector<unsigned>& inRows,
                 unsigned numHypos,
                 const mblas::Vector<unsigned> &outRows)
{
  out.NewSize(numHypos, in.dim(1));
  //cerr << "Assemble=" << out.Debug() << " " << in.Debug() << indices.size() << endl;

  CopyRows(out, in, inRows, outRows);
  return out;
}

__global__ void gSlice(MatrixWrapper<float> out,
                      const MatrixWrapper<float> in,
                       unsigned n, unsigned dim)
{
  unsigned row = blockIdx.x;

  unsigned inCol = threadIdx.x + dim * n;
  unsigned outCol = threadIdx.x;

  while (outCol < out.GetShape().dim(1)) {
    out(row, outCol) = in(row, inCol);

    inCol += blockDim.x;
    outCol += blockDim.x;
  }

}

Matrix& Slice(Matrix& Out,
              const Matrix& In,
              unsigned n, unsigned dim)
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

  unsigned threads = std::min(MAX_THREADS, dim);
  unsigned blocks = In.dim(0);

  gSlice<<<blocks, threads, 0, CudaStreamHandler::GetStream()>>>
    (outWrap, inWrap, n, dim);
  return Out;
}

Matrix& Prod(cublasHandle_t handle, Matrix& C, const Matrix& A, const Matrix& B, bool transB)
{
  //BEGIN_TIMER("Prod");
  assert((A.dim(2) == A.dim(3) == 1) || (B.dim(2) == B.dim(3) == 1));

  Matrix::value_type alpha = 1.0;
  Matrix::value_type beta = 0.0;

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
  //PAUSE_TIMER("Prod");
  return C;
}

Matrix& Prod(Matrix& C, const Matrix& A, const Matrix& B,
             bool transB) {

  //std::cerr << "1C=" << C.Debug() << std::endl;
  //std::cerr << "1A=" << A.Debug() << std::endl;
  //std::cerr << "1B=" << B.Debug() << std::endl;

  Matrix &ret = Prod(CublasHandler::GetHandle(), C, A, B, transB);

  //std::cerr << "2C=" << C.Debug() << std::endl;
  return ret;
}

__global__ void gSoftMax(MatrixWrapper<float> out,
                         const VectorWrapper<unsigned> hypo2BatchWrap,
                         const VectorWrapper<unsigned> sentenceLengthsWrap,
                         unsigned shareSize)
{
  extern __shared__ float _share[];

  unsigned numHypos = out.GetShape().dim(0);
  unsigned maxLength = out.GetShape().dim(1);

  unsigned hypoInd =  blockIdx.x;
  unsigned origSrcPos = threadIdx.x;

  while (hypoInd < numHypos) {
    VectorWrapper<float> _max(_share, shareSize);
    _max[origSrcPos] = out(hypoInd, origSrcPos);
    for (unsigned tid = 0; tid < maxLength; tid += blockDim.x) {
      unsigned srcPos = tid + origSrcPos;
      if (srcPos < maxLength) {
        float value = out(hypoInd, srcPos);

        unsigned batch = hypo2BatchWrap[hypoInd];
        value *= srcPos < sentenceLengthsWrap[batch] ? 1 : 0;
        if (value > _max[origSrcPos]) {
          _max[origSrcPos] = value;
        }
      }
    }

    unsigned len = blockDim.x;
    while (len != 1) {
      __syncthreads();

      unsigned skip = (len + 1) >> 1;
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
    VectorWrapper<float> _sum(_share, shareSize);

    _sum[origSrcPos] = 0.0f;
    for (unsigned tid = 0; tid < maxLength; tid += blockDim.x) {
      unsigned srcPos = tid + origSrcPos;
      if (srcPos < maxLength) {
        out(hypoInd, srcPos) = __expf(out(hypoInd, srcPos) - max);

        unsigned batch = hypo2BatchWrap[hypoInd];
        out(hypoInd, srcPos) *= srcPos < sentenceLengthsWrap[batch] ? 1 : 0; // sentencesMappingWrap(srcPos, batch, 0, 0);
        _sum[origSrcPos] += out(hypoInd, srcPos);
      }
    }

    __syncthreads();

    len = blockDim.x;
    while (len != 1) {
      __syncthreads();

      unsigned skip = (len + 1) >> 1;
      if (origSrcPos < (len >> 1)) {
        _sum[origSrcPos] += _sum[origSrcPos + skip];
      }
      len = (len + 1) >> 1;
    }

    __syncthreads();

    for (unsigned tid = 0; tid < maxLength; tid += blockDim.x) {
      unsigned srcPos = tid + origSrcPos;
      if (srcPos < maxLength) {
        out(hypoInd, srcPos) /= _sum[0];
      }
    }
    __syncthreads();
    hypoInd += gridDim.x;
  }
}

Matrix& Softmax(Matrix& Out,
                const mblas::Vector<unsigned>& hypo2Batch,
                const mblas::Vector<unsigned> &sentenceLengths,
                unsigned batchSize)
{
  unsigned maxLength = Out.dim(1);

  MatrixWrapper<float> outWrap(Out);
  const VectorWrapper<unsigned> hypo2BatchWrap(hypo2Batch);
  const VectorWrapper<unsigned> sentenceLengthsWrap(sentenceLengths);

  unsigned blocks = batchSize;
  unsigned threads = std::min(MAX_THREADS, maxLength);
  unsigned shared = sizeof(float) * threads;

  gSoftMax<<<blocks, threads, shared, CudaStreamHandler::GetStream()>>>
    (outWrap, hypo2BatchWrap, sentenceLengthsWrap, threads);

  return Out;
}

__global__ void gLogSoftMax(MatrixWrapper<float> out, unsigned shareSize)
{
  extern __shared__ float _share[];

  unsigned rows = out.GetShape().dim(0);
  unsigned cols = out.GetShape().dim(1);

  unsigned rowIdx =  blockIdx.x;

  while (rowIdx < rows) {
    //float* _max = _share;
    VectorWrapper<float> _max(_share, shareSize);

    _max[threadIdx.x] = out(rowIdx, threadIdx.x);
    for (unsigned tid = 0; tid < cols; tid += blockDim.x) {
      unsigned id = tid + threadIdx.x;
      if (id < cols) {
        const float &val = out(rowIdx, id);
        if (val > _max[threadIdx.x]) {
          _max[threadIdx.x] = val;
        }
      }
    }

    unsigned len = blockDim.x;
    while (len != 1) {
      __syncthreads();

      unsigned skip = (len + 1) >> 1;
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
    for (unsigned tid = 0; tid < cols; tid += blockDim.x) {
      unsigned id = tid + threadIdx.x;
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

      unsigned skip = (len + 1) >> 1;
      if (threadIdx.x < (len >> 1)) {
        _sum[threadIdx.x] += _sum[threadIdx.x + skip];
      }
      len = (len + 1) >> 1;
    }

    __syncthreads();

    for (unsigned tid = 0; tid < cols; tid += blockDim.x) {
      unsigned id = tid + threadIdx.x;
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


Matrix& LogSoftmax(Matrix& Out)
{
  MatrixWrapper<float> outWrap(Out);

  unsigned blocks = std::min(MAX_BLOCKS, Out.dim(0));
  unsigned threads = std::min(MAX_THREADS, Out.dim(1));
  unsigned shared = sizeof(float) * threads;

  gLogSoftMax<<<blocks, threads, shared, CudaStreamHandler::GetStream()>>>
    (Out, threads);

  return Out;
}

__global__ void gSetColumn(MatrixWrapper<float> in, unsigned noColumn, float value) {
  unsigned n_rows = in.GetShape().dim(0);

  unsigned rowNumber = threadIdx.x  + blockDim.x * blockIdx.x;

  if (rowNumber < n_rows) {
    in(rowNumber, noColumn) = value;
  }
}

void SetColumn(Matrix& In, unsigned noColumn, float value) {
  unsigned nRows = In.dim(0);
  unsigned nBlocks = nRows / MAX_THREADS + ((nRows % MAX_THREADS == 0) ?  0 : 1);
  unsigned nThreads = std::min(MAX_THREADS, nRows);

  MatrixWrapper<float> inWrap(In);

  gSetColumn<<<nBlocks, nThreads, 0, mblas::CudaStreamHandler::GetStream()>>>
    (inWrap, noColumn, value);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void gFill(MatrixWrapper<float> in, float val) {
  unsigned index = threadIdx.x + blockDim.x * blockIdx.x;
  if (index < in.GetShape().size()) {
    in[index] = val;
  }
}

void Fill(Matrix& In, float value) {
  unsigned size = In.size();

  if (value) {
    unsigned nThreads = std::min(MAX_THREADS, size);
    unsigned nBlocks = (size / nThreads) + ((size % nThreads == 0) ? 0 : 1);

    MatrixWrapper<float> inWrap(In);

    gFill<<<nBlocks, nThreads, 0, CudaStreamHandler::GetStream()>>>
      (inWrap, value);
  }
  else {
    HANDLE_ERROR(cudaMemsetAsync(In.data(), 0, size * sizeof(float), CudaStreamHandler::GetStream()));
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void gFill0(MatrixWrapper<float> in,
                      float val,
                      const mblas::VectorWrapper<unsigned> d_newHypoIds,
                      const Shape shape)
{
  unsigned id = threadIdx.x + blockDim.x * blockIdx.x;
  if (id < shape.size()) {
    unsigned dim[SHAPE_SIZE];
    shape.id2Indices(id, dim);

    unsigned hypoInd = d_newHypoIds[dim[0]];
    in(hypoInd, dim[1]) = val;
  }
}

void Fill0(Matrix& in, float value, const mblas::Vector<unsigned> &d_newHypoIds)
{
  //cerr << "in=" << in.Debug(0) << endl;
  //cerr << "d_newHypoIds=" << d_newHypoIds.Debug(2) << endl;

  Shape shape(d_newHypoIds.size(), in.dim(1), 1, 1);
  unsigned size = shape.size();

  unsigned nThreads = std::min(MAX_THREADS, size);
  unsigned nBlocks = (size / nThreads) + ((size % nThreads == 0) ? 0 : 1);

  gFill0<<<nBlocks, nThreads, 0, CudaStreamHandler::GetStream()>>>
    (in, value, d_newHypoIds, shape);
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__
void gMapMatrix(MatrixWrapper<float> in,
                const VectorWrapper<unsigned> sentenceLengthsWrap,
                unsigned i)
{
  unsigned tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < in.GetShape().size()) {
    unsigned numCols = in.GetShape().dim(1);
    unsigned batchIdx = tid / numCols;
    unsigned col = tid % numCols;

    //in[tid] *= mappingWrap(i, batchIdx, 0, 0);
    in(batchIdx, col) *= (i < sentenceLengthsWrap[batchIdx] ? 1 : 0);
  }
}

void MapMatrix(Matrix& state,
              const mblas::Vector<unsigned> &sentenceLengths,
              unsigned i)
{
  // blank out rows in the state matrix where the word position i does not exist
  // mapping is a concatenated array of 1 & 0 of each sentence in the batch to say whether word exists or not.

  unsigned batchSize = state.dim(0);
  unsigned stateLength = state.dim(1);

  unsigned numThreads = std::min(state.size(), MAX_THREADS);
  unsigned numBlocks = (state.size() / numThreads) + ((state.size() % numThreads == 0) ? 0 : 1);

  MatrixWrapper<float> stateWrap(state);
  VectorWrapper<unsigned> sentenceLengthsWrap(sentenceLengths);

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

__device__ unsigned getIndex(const dim3 &dim, const dim3 &val)
{
  unsigned ret = dim.x * val.x + dim.y * val.y + dim.z * val.z;
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

  unsigned cols = in.GetShape().dim(1);

  assert(blockIdx.x < in.GetShape().dim(0));
  assert(blockIdx.y < in.GetShape().dim(2));
  assert(blockIdx.z < in.GetShape().dim(3));

  float* _sum = _share + blockDim.x;
  _sum[threadIdx.x] = 0.0f;
  for (unsigned tid = 0; tid < cols; tid += blockDim.x) {
    unsigned id = tid + threadIdx.x;
    if (id < cols) {
      _sum[threadIdx.x] += in(blockIdx.x, id, blockIdx.y, blockIdx.z);
    }
  }
  __syncthreads();
  unsigned len = blockDim.x;
  while(len != 1) {
    __syncthreads();
    unsigned skip = (len + 1) >> 1;
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
  for (unsigned tid = 0; tid < cols; tid += blockDim.x) {
    unsigned id = tid + threadIdx.x;
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
    unsigned skip = (len + 1) >> 1;
    if(threadIdx.x < (len >> 1))
      _sqSum[threadIdx.x] += _sqSum[threadIdx.x + skip];
    len = (len + 1) >> 1;
  }
  __syncthreads();
  float sigma = sqrtf(eps + (_sqSum[0] / cols));
  __syncthreads();

  for (unsigned tid = 0; tid < cols; tid += blockDim.x) {
    unsigned id = tid + threadIdx.x;
    if(id < cols) {
      float &val = out(blockIdx.x, id, blockIdx.y, blockIdx.z);
      if (betaWrap.GetShape().size()) {
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

  unsigned numThreads = std::min(in.dim(1), MAX_THREADS);
  dim3 numBlocks(in.dim(0), in.dim(2), in.dim(3));
  unsigned shared = numThreads * sizeof(float) * 2;

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
#define LOWEST_FLOAT -1111111111111
#define HIGHEST_FLOAT +999999999999

__global__
void gBeamSizeInit(VectorWrapper<unsigned> hypo2BeamSize,
                    VectorWrapper<unsigned> activeBatch2Hypo,
                    VectorWrapper<unsigned> hypo2Candidate,
                    VectorWrapper<unsigned> hypo2NextHypo,
                    VectorWrapper<char> isFirsts,
                    const VectorWrapper<unsigned> beamSizes)
{
  unsigned nextHypoInd = 0;
  unsigned candidateInd = 0;

  unsigned hypoInd = 0, activeBatchInd = 0;
  //printf("beamSizes.size()=%u \n", beamSizes.size());
  for (unsigned batchInd = 0; batchInd < beamSizes.size(); ++batchInd) {
    unsigned beamSize = beamSizes[batchInd];
    /*
    printf("batchInd=%u ", batchInd);
    printf("beamSize=%u ", beamSize);
    printf("hypoInd=%u ", hypoInd);
    printf("activeBatchInd=%u \n", activeBatchInd);
    */
    bool isFirst = isFirsts[batchInd];
    if (beamSize) {
      if (isFirst) {
        //printf("hypoInd=%i hypo2BeamSize=%i \n", hypoInd, hypo2BeamSize.size());
        assert(hypoInd < hypo2BeamSize.size());
        assert(hypoInd < hypo2Candidate.size());
        assert(activeBatchInd < activeBatch2Hypo.size());
        activeBatch2Hypo[activeBatchInd] = hypoInd;
        hypo2BeamSize[hypoInd] = beamSize;
        hypo2Candidate[hypoInd] = candidateInd;
        hypo2NextHypo[hypoInd] = nextHypoInd;


        ++hypoInd;
        candidateInd += beamSize;
      }
      else {
        assert(activeBatchInd < activeBatch2Hypo.size());
        activeBatch2Hypo[activeBatchInd] = hypoInd;

        for (unsigned j = 0; j < beamSize; ++j) {
          assert(hypoInd < hypo2BeamSize.size());
          assert(hypoInd < hypo2Candidate.size());
          hypo2BeamSize[hypoInd] = beamSize;
          hypo2Candidate[hypoInd] = candidateInd;
          hypo2NextHypo[hypoInd] = nextHypoInd;
          ++hypoInd;

          candidateInd += beamSize;
        }
      }

      ++activeBatchInd;
      nextHypoInd += beamSize;
    }
  }

  //printf("hypoInd=%i \n", hypoInd);
  //printf("activeBatchInd=%i \n", activeBatchInd);
}

__device__
float GetMaxScore(const MatrixWrapper<NthOutBatch> &nBestMatrix)
{
  float ret = LOWEST_FLOAT;
  for (unsigned i = 0; i < nBestMatrix.GetShape().dim(1); ++i) {
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
  assert(arrSize <= vec.size());

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
                  VectorWrapper<NthOutBatch> &row,
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
    MergeElement(minScore, row, arrSize, ele);

    /*
    printf("arrInd=%d ind=%d vocabId=%d \n",
          arrInd,
          _max[threadIdx.x].ind,
          _max[threadIdx.x].vocabId);
    */
  }
}

__device__
void NBestAndMax(VectorWrapper<NthOutBatch> &nBestCandidates,
              float &topScore,
              const MatrixWrapper<float> &in,
              const MatrixWrapper<float> &b4,
              unsigned hypoInd,
              unsigned maxBeamSize,
              bool forbidUNK,
              const VectorWrapper<unsigned> &hypo2BeamSize,
              const VectorWrapper<unsigned> &hypo2Candidate)
{
  extern __shared__ char _sharePtr[];

  // placeholder for shared mem in subsequent function SumAndLogSoftMax
  //MatrixWrapper<float> maxMatrix((float*)_sharePtr, blockDim.x, 1, 1, 1);

  void *ptrOffset = _sharePtr + sizeof(float) * blockDim.x;
  MatrixWrapper<NthOutBatch> nBestMatrix((NthOutBatch*)ptrOffset, blockDim.x, maxBeamSize, 1, 1);
  VectorWrapper<NthOutBatch> row = nBestMatrix.Row(threadIdx.x);

  unsigned vocabSize = in.GetShape().dim(1);

  assert(hypoInd < hypo2BeamSize.size());
  unsigned beamSize = hypo2BeamSize[hypoInd];

  float minScore = HIGHEST_FLOAT;

  // init
  unsigned vocabInd = threadIdx.x;
  unsigned i = 0;
  while (vocabInd < vocabSize && i < beamSize) {
    const float score = in(hypoInd, vocabInd) + b4(0, vocabInd);

    unsigned arrInd = hypoInd * vocabSize + vocabInd;
    NthOutBatch ele(arrInd, score, hypoInd, vocabInd);

    AddElement(minScore, i, row, forbidUNK, vocabInd, ele);

    vocabInd += blockDim.x;
  }

  // MAIN LOOP
  while (vocabInd < vocabSize) {
    const float score = in(hypoInd, vocabInd) + b4(0, vocabInd);
    unsigned arrInd = hypoInd * vocabSize + vocabInd;
    NthOutBatch ele(arrInd, score, hypoInd, vocabInd);

    MergeElement(minScore, row, beamSize, ele, forbidUNK, vocabInd);

    vocabInd += blockDim.x;
  } // while (vocabInd < vocabSize) {

  // merge nbest from different threads
  unsigned len = blockDim.x;
  while (len != 1) {
    __syncthreads();
    unsigned skip = (len + 1) >> 1;
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

  if (threadIdx.x == 0) {
    __syncthreads();

    // copy to output array
    assert(hypoInd < hypo2Candidate.size());
    unsigned candidateInd = hypo2Candidate[hypoInd];
    for (unsigned i = 0; i < beamSize; ++i) {
      const NthOutBatch &curr = nBestMatrix(0, i);

      //printf("candidateInd=%u nBestCandidates=%u \n", candidateInd, nBestCandidates.size());
      assert(candidateInd + i < nBestCandidates.size());
      nBestCandidates[candidateInd + i] = curr;
    }
  }

  __syncthreads();
  topScore = GetMaxScore(nBestMatrix);

}

///////////////////////////////////////////////////////////////////////////////////////////////////////
__device__
void SumAndLogSoftMax(VectorWrapper<NthOutBatch> &nBestCandidates,
                            const MatrixWrapper<float> &in,
                            const MatrixWrapper<float> &b4,
                            unsigned hypoInd,
                            unsigned maxBeamSize,
                            float topScore,
                            const VectorWrapper<unsigned> &hypo2BeamSize,
                            const VectorWrapper<unsigned> &hypo2Candidate)
{
  extern __shared__ float _share[];
  VectorWrapper<float> _sum(_share, blockDim.x);

  unsigned vocabSize = in.GetShape().dim(1);

  // calc sum
  _sum[threadIdx.x] = 0.0f;
  for (unsigned id = threadIdx.x; id < vocabSize; id += blockDim.x) {
    //row[id] = exp(row[id] - max);
    float val = in(hypoInd, id) + b4(0, id);
    val = __expf(val - topScore);
    _sum[threadIdx.x] += val;
  }

  unsigned len = blockDim.x;
  while (len != 1) {
    __syncthreads();

    unsigned skip = (len + 1) >> 1;
    if (threadIdx.x < (len >> 1)) {
      _sum[threadIdx.x] += _sum[threadIdx.x + skip];
    }
    len = (len + 1) >> 1;
  }

  // apply partition and log to top
  if (threadIdx.x == 0) {
    __syncthreads();
    //printf("val=%f %f \n", in(rowIdx, ele.vocabId, 0, 0), val);

    // nbest
    unsigned beamSize = hypo2BeamSize[hypoInd];
    unsigned startPos = hypo2Candidate[hypoInd];
    for (unsigned i = 0; i < beamSize; ++i) {
      //__syncthreads();
      NthOutBatch &ele = nBestCandidates[startPos + i];

      float &val = ele.score;
      val = __expf(val - topScore);
      val = __logf(val /_sum[0]);
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void gLogSoftMax(VectorWrapper<NthOutBatch> nBestCandidates,
                        const MatrixWrapper<float> in,
                        const MatrixWrapper<float> b4,
                        unsigned maxBeamSize,
                        bool forbidUNK,
                        const VectorWrapper<unsigned> hypo2BeamSize,
                        const VectorWrapper<unsigned> hypo2Candidate)
{
  unsigned hypos = in.GetShape().dim(0);
  unsigned vocabSize = in.GetShape().dim(1);

  unsigned hypoInd =  blockIdx.x; // index of previous hypo
  while (hypoInd < hypos) {
    float topScore;

    NBestAndMax(nBestCandidates,
            topScore,
            in,
            b4,
            hypoInd,
            maxBeamSize,
            forbidUNK,
            hypo2BeamSize,
            hypo2Candidate);

    //__syncthreads();

    SumAndLogSoftMax(nBestCandidates,
                in,
                b4,
                hypoInd,
                maxBeamSize,
                topScore,
                hypo2BeamSize,
                hypo2Candidate);


    __syncthreads();
    hypoInd += gridDim.x;
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void gNBestPerBatch(VectorWrapper<NthOutBatch> nBest,
                        VectorWrapper<NthOutBatch> nBestCandidates,
                        const MatrixWrapper<float> in,
                        const VectorWrapper<float> costs,
                        unsigned maxBeamSize,
                        bool forbidUNK,
                        VectorWrapper<char> isFirsts,
                        const VectorWrapper<unsigned> hypo2BeamSize,
                        const VectorWrapper<unsigned> activeBatch2Hypo,
                        const VectorWrapper<unsigned> hypo2Candidate,
                        const VectorWrapper<unsigned> hypo2NextHypo)
{
  //printf("start gNBestPerBatch\n");
  //unsigned rows = in.GetShape().dim(0);
  unsigned activeBatchSize = activeBatch2Hypo.size();

  unsigned batchInd =  blockIdx.x;
  while (batchInd < activeBatchSize) {
    assert(batchInd < activeBatch2Hypo.size());
    assert(batchInd < hypo2BeamSize.size());
    assert(batchInd < nBest.size());

    unsigned hypoInd = activeBatch2Hypo[batchInd];
    unsigned beamSize = hypo2BeamSize[hypoInd];
    assert(beamSize);

    unsigned nextHypoInd = hypo2NextHypo[hypoInd];
    bool isFirst = isFirsts[batchInd];

    // candiate from 1st hypo
    float minScore = HIGHEST_FLOAT;
    assert(hypoInd < hypo2Candidate.size());
    unsigned candidateInd = hypo2Candidate[hypoInd];
    for (unsigned i = 0; i < beamSize; ++i) {
      //printf("prevHypoInd=%, candidateInd=%d \n", prevHypoInd, candidateInd);
      assert(hypoInd < costs.size());
      float prevCost = costs[hypoInd];

      assert((nextHypoInd + i) < nBest.size());
      assert(candidateInd + i < nBestCandidates.size());
      const NthOutBatch &candidate = nBestCandidates[candidateInd + i];
      nBest[nextHypoInd + i] = candidate;

      float &score = nBest[nextHypoInd + i].score;
      score += prevCost;

      if (score < minScore) {
        minScore = score;
      }
    }

    // candidates from other previous hypos
    if (!isFirst) {
      assert(nextHypoInd < nBest.size());
      VectorWrapper<NthOutBatch> offset = nBest.Offset(nextHypoInd);

      for (unsigned hypoOffset = 1; hypoOffset < beamSize; ++hypoOffset) {
        //printf("hypoInd=%d \n", (hypoInd + hypoOffset));

        //printf("prevHypoInd=%, candidateInd=%d \n", prevHypoInd, candidateInd);
        assert((hypoInd + hypoOffset) < costs.size());
        float prevCost = costs[hypoInd + hypoOffset];

        assert((hypoInd + hypoOffset) < hypo2Candidate.size());
        unsigned candidateInd = hypo2Candidate[hypoInd + hypoOffset];

        for (unsigned candidateOffset = 0; candidateOffset < beamSize; ++candidateOffset) {
          assert((candidateInd + candidateOffset) < nBestCandidates.size());
          NthOutBatch &candidate = nBestCandidates[candidateInd + candidateOffset];
          candidate.score += prevCost;

          if (candidate.score > minScore) {
            MergeElement(minScore, offset, beamSize, candidate);
          }
        }
      }
    }

    batchInd += gridDim.x;
  }
  //printf("end gNBestPerBatch\n");
}

///////////////////////////////////////////////////////////////////////////////////////////////////////

void LogSoftmaxAndNBest(mblas::Vector<NthOutBatch> &nBest,
                const Matrix& in,
                const Matrix& b4,
                const mblas::Vector<float> &costs,
                const Histories& histories,
                bool forbidUNK,
                unsigned maxBeamSize)
{
  //BEGIN_TIMER("LogSoftmax excl kernels");
  //HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
  //cerr << "LogSoftmaxAndNBest0" << endl;

  //cerr << "in=" << in.Debug(0) << endl;
  //cerr << "histories=" << histories.Debug(1) << endl;
  //cout << "histories=" << histories.Debug(1) << endl;

  std::vector<char> isFirsts = histories.IsFirsts();

  // create beam size vectors on GPU but exclude empty beams
  unsigned candidateInd = histories.NumCandidates();
  unsigned activeBatchSize = histories.NumActive();
  unsigned numHypos = in.dim(0);
  unsigned numNextHypos = histories.GetTotalBeamSize();

  mblas::Vector<char> d_isFirsts(isFirsts);
  mblas::Vector<unsigned> d_beamSizes(histories.GetBeamSizes());
  mblas::Vector<unsigned> hypo2BeamSize(numHypos);
  mblas::Vector<unsigned> hypo2Candidate(numHypos);
  mblas::Vector<unsigned> hypo2NextHypo(numHypos);
  mblas::Vector<unsigned> activeBatch2Hypo(activeBatchSize);
  mblas::Vector<NthOutBatch> nBestCandidates(candidateInd);
  //PAUSE_TIMER("LogSoftmax excl kernels");

  //cerr << "LogSoftmaxAndNBest1" << endl;

  //BEGIN_TIMER("gBeamSizeInit");
  gBeamSizeInit<<<1, 1, 0, CudaStreamHandler::GetStream()>>>
    (hypo2BeamSize,
    activeBatch2Hypo,
    hypo2Candidate,
    hypo2NextHypo,
    d_isFirsts,
    d_beamSizes
    );
  //PAUSE_TIMER("gBeamSizeInit");
  HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
  /*
  cerr << "numHypos=" << numHypos << endl;
  cerr << "numNextHypos=" << numNextHypos << endl;
  cerr << "isFirsts=" << Debug(isFirsts, 2) << endl;
  cerr << "in=" << in.Debug(0) << endl;
  cerr << "activeBatchSize=" << activeBatchSize << endl;
  cerr << "candidateInd=" << candidateInd << endl;
  cerr << "hypo2BeamSize=" << hypo2BeamSize.Debug(2) << endl;
  cerr << "hypo2Candidate=" << hypo2Candidate.Debug(2) << endl;
  cerr << "nBest=" << nBest.Debug(2) << endl;
  cerr << "nBestCandidates=" << nBestCandidates.Debug(2) << endl;
  cerr << "histories=" << histories.Debug(2) << endl;
  cerr << "activeBatch2Hypo=" << activeBatch2Hypo.Debug(2) << endl;
  cerr << "hypo2NextHypo=" << hypo2NextHypo.Debug(2) << endl;
  cerr << endl;
  */

  unsigned blocks = std::min(MAX_BLOCKS, numHypos);
  unsigned threads = std::min(MAX_THREADS, in.dim(1));
  unsigned shared = sizeof(NthOutBatch) * threads * maxBeamSize
             + sizeof(float) * threads;
  //HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
  //cerr << "LogSoftmaxAndNBest2" << endl;
  //BEGIN_TIMER("gLogSoftMax");
  gLogSoftMax<<<blocks, threads, shared, CudaStreamHandler::GetStream()>>>
    (nBestCandidates,
     in,
     b4,
     maxBeamSize,
     forbidUNK,
     hypo2BeamSize,
     hypo2Candidate);
  //PAUSE_TIMER("gLogSoftMax");
  //HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
  //cerr << "LogSoftmaxAndNBest3" << endl;

  blocks = std::min(MAX_BLOCKS, activeBatchSize);
  /*
  cerr << "nBestCandidates=" << nBestCandidates.Debug(2) << endl;
  cerr << "d_isFirsts=" << d_isFirsts.Debug(2) << endl;
  cerr << "activeBatch2Hypo=" << activeBatch2Hypo.Debug(2) << endl;
  cerr << "hypo2NextHypo=" << hypo2NextHypo.Debug(2) << endl;
  cerr << "hypo2BeamSize=" << hypo2BeamSize.Debug(2) << endl;
  cerr << "1nBest=" << nBest.Debug(2) << endl;
  */
  //BEGIN_TIMER("gNBestPerBatch");
  gNBestPerBatch<<<blocks, 1, 0, CudaStreamHandler::GetStream()>>>
    (nBest,
     nBestCandidates,
     in,
     costs,
     maxBeamSize,
     forbidUNK,
     d_isFirsts,
     hypo2BeamSize,
     activeBatch2Hypo,
     hypo2Candidate,
     hypo2NextHypo);
  //PAUSE_TIMER("gNBestPerBatch");
  //HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
  //cerr << "LogSoftmaxAndNBest4" << endl;
  //cerr << "2nBest=" << nBest.Debug(2) << endl;

  //HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
  //cerr << "step3" << endl;
  //cerr << "3costs=" << Debug(costs, 0) << endl;
}

__global__
void gUpdateSentenceLengths(const VectorWrapper<unsigned> newSentenceLengths,
                            const VectorWrapper<unsigned> newBatchIds,
                            VectorWrapper<unsigned> sentenceLengths)
{
  unsigned id =  threadIdx.x;
  while (id < newSentenceLengths.size()) {
    unsigned sentenceLength = newSentenceLengths[id];
    unsigned batchId = newBatchIds[id];

    assert(batchId < sentenceLengths.size());
    sentenceLengths[batchId] = sentenceLength;

    id += blockDim.x;
  }
}

void UpdateSentenceLengths(const mblas::Vector<unsigned> &d_newBatchIds,
                          const mblas::Vector<unsigned> &d_newSentenceLengths,
                          mblas::Vector<unsigned> &sentenceLengths)
{
  assert(d_newSentenceLengths.size() == d_newBatchIds.size());
  assert(d_newSentenceLengths.size() <= sentenceLengths.size());

  unsigned blocks = 1;
  unsigned threads = std::min(MAX_THREADS, d_newSentenceLengths.size());

  //cerr << "1sentenceLengths=" << sentenceLengths.Debug(2) << endl;
  gUpdateSentenceLengths<<<blocks, threads, 0, CudaStreamHandler::GetStream()>>>(d_newSentenceLengths, d_newBatchIds, sentenceLengths);

  //HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
  //cerr << "2sentenceLengths=" << sentenceLengths.Debug(2) << endl;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__
void gAddNewData3(mblas::MatrixWrapper<float> dest,
                const mblas::MatrixWrapper<float> source,
                unsigned batchId,
                unsigned newSentenceOffset,
                const Shape shape)
{
  unsigned id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id < shape.size()) {
    unsigned dim0 = id / shape.dim(1);
    unsigned dim1 = id % shape.dim(1);

    dest(dim0, dim1, 0, batchId) = source(dim0, dim1, 0, newSentenceOffset);
  }
}

void AddNewSourceContext(mblas::Matrix &matrix,
                const vector<unsigned> &newBatchIds,
                const std::vector<BufferOutput> &newSentences)
{
  //BEGIN_TIMER("AddNewSourceContext");
  //cerr << "matrix=" << matrix.Debug(0) << endl;
  //cerr << "newBatchIds=" << amunmt::Debug(newBatchIds, 2) << endl;

  for (unsigned i = 0; i < newSentences.size(); ++i) {
    const BufferOutput &eleSent = newSentences[i];
    const EncOutPtr &encOut = eleSent.GetEncOut();
    const mblas::Matrix &newMatrix = encOut->Get<EncOutGPU>().GetSourceContext();
    //cerr << "sourceContext=" << sourceContext.Debug(1) << endl;
    //cerr << "newMatrix=" << newMatrix.Debug(1) << endl;

    unsigned batchId = newBatchIds[i];
    unsigned newSentenceOffset = eleSent.GetSentenceOffset();
    const SentencePtr &sent = encOut->GetSentences().Get(newSentenceOffset);
    unsigned sentLength = sent->size();
    //cerr << "batchId=" << batchId << endl;
    //cerr << "newSentenceOffset=" << newSentenceOffset << endl;

    Shape shape(sentLength, newMatrix.dim(1), 1, 1);

    assert(batchId < matrix.dim(3));
    assert(newSentenceOffset < newMatrix.dim(3));
    //assert(matrix.dim(0) >= newMatrix.dim(0));
    assert(matrix.dim(1) == newMatrix.dim(1));
    assert(matrix.dim(2) == newMatrix.dim(2) == 1);

    unsigned size = shape.size();
    unsigned threads = std::min(MAX_THREADS, size);
    unsigned blocks  = size / threads + ((size % threads == 0) ?  0 : 1);

    gAddNewData3<<<blocks, threads, 0, CudaStreamHandler::GetStream()>>>(matrix, newMatrix, batchId, newSentenceOffset, shape);
  }

  //PAUSE_TIMER("AddNewSourceContext");
}

void AddNewSCU(mblas::Matrix &matrix,
                const vector<unsigned> &newBatchIds,
                const std::vector<BufferOutput> &newSentences)
{
  //BEGIN_TIMER("AddNewSCU");
  //cerr << "sourceContext=" << sourceContext.Debug(0) << endl;

  for (unsigned i = 0; i < newSentences.size(); ++i) {
    const BufferOutput &eleSent = newSentences[i];
    const EncOutPtr &encOut = eleSent.GetEncOut();
    const mblas::Matrix &newMatrix = encOut->Get<EncOutGPU>().GetSCU();
    //cerr << "sourceContext=" << sourceContext.Debug(1) << endl;
    //cerr << "newMatrix=" << newMatrix.Debug(1) << endl;

    unsigned batchId = newBatchIds[i];
    unsigned newSentenceOffset = eleSent.GetSentenceOffset();
    const SentencePtr &sent = encOut->GetSentences().Get(newSentenceOffset);
    unsigned sentLength = sent->size();
    //cerr << "batchId=" << batchId << endl;
    //cerr << "newSentenceOffset=" << newSentenceOffset << endl;

    Shape shape(sentLength, newMatrix.dim(1), 1, 1);

    assert(batchId < matrix.dim(3));
    assert(newSentenceOffset < newMatrix.dim(3));
    //assert(matrix.dim(0) >= newMatrix.dim(0));
    assert(matrix.dim(1) == newMatrix.dim(1));
    assert(matrix.dim(2) == newMatrix.dim(2) == 1);

    unsigned size = shape.size();
    unsigned threads = std::min(MAX_THREADS, size);
    unsigned blocks  = size / threads + ((size % threads == 0) ?  0 : 1);

    gAddNewData3<<<blocks, threads, 0, CudaStreamHandler::GetStream()>>>(matrix, newMatrix, batchId, newSentenceOffset, shape);
  }

  //PAUSE_TIMER("AddNewSCU");
}

//////////////////////////////////////////////////////////////////////////////////////////
__global__
void gAddNewData0(mblas::MatrixWrapper<float> dest,
                const mblas::MatrixWrapper<float> source,
                unsigned batchId,
                unsigned newSentenceOffset)
{
  unsigned id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id < dest.GetShape().dim(1)) {
    dest(batchId, id, 0, 0) = source(newSentenceOffset, id, 0, 0);
  }
}

void AddNewStates(mblas::Matrix& matrix,
                  const mblas::Matrix &newMatrix,
                  unsigned hypoId,
                  unsigned newSentenceOffset,
                  const CellState &newState)
{
  assert(hypoId < matrix.dim(0));
  assert(newSentenceOffset < newMatrix.dim(0));
  assert(matrix.dim(1) == newMatrix.dim(1));
  assert(matrix.dim(2) == newMatrix.dim(2) == 1);
  assert(matrix.dim(3) == newMatrix.dim(3) == 1);

  unsigned size = newMatrix.dim(1);
  unsigned threads = std::min(MAX_THREADS, size);
  unsigned blocks  = size / threads + ((size % threads == 0) ?  0 : 1);

  gAddNewData0<<<blocks, threads, 0, CudaStreamHandler::GetStream()>>>(matrix, newMatrix, hypoId, newSentenceOffset);

}

void AddNewStates(CellState& State,
                const std::vector<unsigned> &newHypoIds,
                const std::vector<BufferOutput> &newSentences)
{
  //BEGIN_TIMER("AddNewStates");
  //cerr << "State=" << State.Debug(0) << endl;
  //cerr << "newHypoIds=" << amunmt::Debug(newHypoIds, 2) << endl;

  for (unsigned i = 0; i < newSentences.size(); ++i) {
    const BufferOutput &eleSent = newSentences[i];
    const EncOutPtr &encOut = eleSent.GetEncOut();
    const CellState &newState = encOut->Get<EncOutGPU>().GetCellState();
    //cerr << "newState=" << newState.Debug(1) << endl;

    unsigned hypoId = newHypoIds[i];
    unsigned newSentenceOffset = eleSent.GetSentenceOffset();
    //cerr << "hypoId=" << hypoId << endl;
    //cerr << "newSentenceOffset=" << newSentenceOffset << endl;

    AddNewStates(*State.output, *newState.output, hypoId, newSentenceOffset, newState);
  }

  //PAUSE_TIMER("AddNewStates");
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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
