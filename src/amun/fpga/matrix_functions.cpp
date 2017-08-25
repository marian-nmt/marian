#include <cassert>
#include "matrix_functions.h"
#include "matrix.h"
#include "kernel.h"
#include "array.h"
#include "model.h"

using namespace std;

namespace amunmt {
namespace FPGA {
namespace mblas {

//////////////////////////////////////////////////////////////////////////////////////
template <typename T>
void SetKernelArg(cl_kernel kernel, cl_uint argNum, const T &t)
{
  //std::cerr << "arg" << argNum << "=" << t << std::endl ;
  CheckError( clSetKernelArg(kernel, argNum, sizeof(T), &t) );
}

template<typename T, typename... Args>
void SetKernelArg(cl_kernel kernel, cl_uint argNum, const T &t, Args... args) // recursive variadic function
{
  //std::cerr << "arg" << argNum << "=" << t << std::endl ;
  CheckError( clSetKernelArg(kernel, argNum, sizeof(T), &t) );

  SetKernelArg(kernel, argNum + 1, args...) ;
}

template<typename... Args>
void CallOpenCL(
    const std::string &filePath,
    const std::string &kernelName,
    const OpenCLInfo &openCLInfo,
    Args... args
    )
{
  cl_int err;
  size_t global;                      // global domain size for our calculation
  size_t local;                       // local domain size for our calculation

  cl_mem output = clCreateBuffer(openCLInfo.context, CL_MEM_WRITE_ONLY, sizeof(size_t), NULL, &err);
  CheckError(err);
  assert(output);

  // create kernel
  cl_kernel kernel = CreateKernel(filePath, kernelName, openCLInfo);

  // Set the arguments to our compute kernel
  SetKernelArg(kernel, 0, args...);

  // Get the maximum work group size for executing the kernel on the device
  //
  CheckError( clGetKernelWorkGroupInfo(kernel, openCLInfo.device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL) );

  //global = 1024;
  local = 1;
  global = 1;

  //cerr << "local=" << local << endl;
  //cerr << "global=" << global << endl;

  CheckError( clEnqueueNDRangeKernel(openCLInfo.commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL) );

  // Wait for the command commands to get serviced before reading back results
  //
  CheckError( clFinish(openCLInfo.commands) );

}

//////////////////////////////////////////////////////////////////////////////////////

float SumFloat(
    const OpenCLInfo &openCLInfo,
    const cl_mem &mem,
    uint size)
{
  cl_int err;
  cl_mem output = clCreateBuffer(openCLInfo.context, CL_MEM_WRITE_ONLY, sizeof(float), NULL, &err);
  CheckError(err);
  assert(output);

  CallOpenCL("kernels/matrix_functions.cl", "sum_float", openCLInfo,
      mem, output, size);

  // Read back the results from the device to verify the output
  //
  float results;
  CheckError( clEnqueueReadBuffer( openCLInfo.commands, output, CL_TRUE, 0, sizeof(float), &results, 0, NULL, NULL ) );
  return results;
}

uint SumUInt(
    const OpenCLInfo &openCLInfo,
    const cl_mem &mem,
    uint size)
{
  cl_int err;
  cl_mem output = clCreateBuffer(openCLInfo.context, CL_MEM_WRITE_ONLY, sizeof(uint), NULL, &err);
  CheckError(err);
  assert(output);

  CallOpenCL("kernels/matrix_functions.cl", "sum_uint", openCLInfo,
      mem, output, size);

  // Read back the results from the device to verify the output
  //
  uint results;
  CheckError( clEnqueueReadBuffer( openCLInfo.commands, output, CL_TRUE, 0, sizeof(uint), &results, 0, NULL, NULL ) );
  return results;
}

Matrix& Copy(Matrix& Out, const Matrix& In)
{
  //cerr << "Out=" << Out.Debug() << endl;
  const OpenCLInfo &openCLInfo = In.GetOpenCLInfo();

  Out.Resize(In.dim(0), In.dim(1), In.dim(2), In.dim(3));

  CheckError( clEnqueueCopyBuffer(openCLInfo.commands, In.data(), Out.data(), 0, 0, sizeof(float) * In.size(), 0, NULL, NULL) );

}

Matrix& CopyRows(
	Matrix& Out,
	const Matrix& In,
	const Array<uint>& indices)
{
  //cerr << "Out=" << Out.Debug() << endl;
  //cerr << "indices=" << indices.Debug() << endl;
  const OpenCLInfo &openCLInfo = In.GetOpenCLInfo();

  CallOpenCL("kernels/matrix_functions.cl", "gCopyRows", openCLInfo,
      Out.data(),
      In.data(),
      In.dimUInt(1),
      indices.data(),
      indices.sizeUInt());

  return Out;
}

Matrix& Assemble(
		Matrix& Out,
		 const Matrix& In,
		 const Array<uint>& indices)
{
  //cerr << "indices=" << indices.Debug(true) << endl;
  const OpenCLInfo &openCLInfo = In.GetOpenCLInfo();

  Out.Resize(indices.size(), In.dim(1));
  CopyRows(Out, In, indices);
  return Out;
}

void Fill(
    Matrix& In,
    float value)
{
  const OpenCLInfo &openCLInfo = In.GetOpenCLInfo();

  CheckError( clEnqueueFillBuffer(openCLInfo.commands, In.data(), &value, sizeof(float), 0, In.size() * sizeof(float), 0, NULL, NULL) );
  CheckError( clFinish(openCLInfo.commands) );
}

Matrix& Transpose(Matrix& Out, const Matrix& In)
{
  const OpenCLInfo &openCLInfo = Out.GetOpenCLInfo();

  Out.Resize(In.dim(1), In.dim(0));

  CallOpenCL("kernels/matrix_functions.cl", "transpose", openCLInfo,
      Out.data(), In.data(), In.dimUInt(0), In.dimUInt(1));

  return Out;
}

Matrix& Transpose(Matrix& Out)
{
  const OpenCLInfo &openCLInfo = Out.GetOpenCLInfo();
  Matrix Temp(openCLInfo);
  Transpose(Temp, Out);
  Out.Swap(Temp);
  return Out;
}

Matrix& Concat(Matrix& Out, const Matrix& In)
{
  const OpenCLInfo &openCLInfo = Out.GetOpenCLInfo();

  size_t oldOutSize = Out.size();
  Out.Resize(Out.dim(0) + In.dim(0), Out.dim(1));

  size_t inSize = In.size();
  //cerr << "Concat=" << inSize << " " << oldOutSize << " " << Out.size() << endl;

  CheckError( clEnqueueCopyBuffer(openCLInfo.commands, In.data(), Out.data(), 0, sizeof(float) * oldOutSize, sizeof(float) * inSize, 0, NULL, NULL) );
  //CheckError( clFinish(openCLInfo.commands) );

  return Out;
}


Matrix& Prod(Matrix& C, const Matrix& A, const Matrix& B,
             bool transA, bool transB)
{
  const OpenCLInfo &openCLInfo = A.GetOpenCLInfo();

  assert(!transA);
  assert(!transB);
  assert(A.dim(1) == B.dim(0));

  C.Resize(A.dim(0), B.dim(1), A.dim(2), A.dim(3));
  //Fill(C, 0);
  //cerr << "C=" << C.Debug(1) << endl;

  // Set the arguments to our compute kernel
  uint rowsA = A.dimUInt(0) * A.dimUInt(2) * A.dimUInt(3);
  uint colsA = A.dimUInt(1);
  uint rowsB = B.dimUInt(0) * B.dimUInt(2) * B.dimUInt(3);
  uint colsB = B.dimUInt(1);

  CallOpenCL("kernels/matrix_functions.cl", "prod", openCLInfo,
      C.data(),
      A.data(),
      B.data(),
      rowsA,
      colsA,
      rowsB,
      colsB);

  return C;
}

void ElementwiseOps(mblas::Matrix& NextState,
                    const mblas::Matrix& State,
                    const mblas::Matrix& RUH,
                    const mblas::Matrix& Temp,
                    const mblas::Matrix& B,
                    const mblas::Matrix& Bx1,
                    const mblas::Matrix& Bx2,
                    const uint &rows,
                    const uint &cols)
{
  const OpenCLInfo &openCLInfo = NextState.GetOpenCLInfo();

  CallOpenCL("kernels/matrix_functions.cl", "gElementwiseOps", openCLInfo,
      NextState.data(),
      State.data(),
      RUH.data(),
      Temp.data(),
      B.data(),
      Bx1.data(),
      Bx2.data(),
      rows,
      cols
      );
}


Matrix& ElementLogit(Matrix& Out, const Matrix& In)
{
  const OpenCLInfo &openCLInfo = Out.GetOpenCLInfo();

  uint rows  = Out.dimUInt(0) * Out.dimUInt(2) * Out.dimUInt(3);
  uint cols = Out.dimUInt(1);

  CallOpenCL("kernels/matrix_functions.cl", "gLogit", openCLInfo,
      Out.data(), In.data(), rows, cols);

  return Out;
}

Matrix& ElementTanh(Matrix& Out, const Matrix& In1, const Matrix& In2)
{
  const OpenCLInfo &openCLInfo = Out.GetOpenCLInfo();

  uint rows  = Out.dimUInt(0) * Out.dimUInt(2) * Out.dimUInt(3);
  uint cols = Out.dimUInt(1);

  CallOpenCL("kernels/matrix_functions.cl", "gElementTanh", openCLInfo,
      Out.data(), In1.data(), In2.data(), rows, cols);

  return Out;
}

Matrix& ElementTanh2(Matrix& Out, const Matrix& In1, const Matrix& In2)
{
  const OpenCLInfo &openCLInfo = Out.GetOpenCLInfo();

  uint rows  = Out.dimUInt(0) * Out.dimUInt(2) * Out.dimUInt(3);
  uint cols = Out.dimUInt(1);

  CallOpenCL("kernels/matrix_functions.cl", "gElementTanh2", openCLInfo,
      Out.data(), In1.data(), In2.data(), rows, cols);

  return Out;
}

Matrix& ElementWhatever(Matrix& Out, const Matrix& In1, const Matrix& In2)
{
  const OpenCLInfo &openCLInfo = Out.GetOpenCLInfo();

  uint rows  = Out.dimUInt(0) * Out.dimUInt(2) * Out.dimUInt(3);
  uint cols = Out.dimUInt(1);

  CallOpenCL("kernels/matrix_functions.cl", "gElementWhatever", openCLInfo,
      Out.data(), In1.data(), In2.data(), rows, cols);

  return Out;
}

Matrix& ElementAddWeighted(Matrix& Out, float weight, const Matrix& In)
{
  const OpenCLInfo &openCLInfo = Out.GetOpenCLInfo();

  uint rows  = Out.dimUInt(0) * Out.dimUInt(2) * Out.dimUInt(3);
  uint cols = Out.dimUInt(1);

  CallOpenCL("kernels/matrix_functions.cl", "gElementAddWeighted", openCLInfo,
      Out.data(), In.data(), rows, cols);

  return Out;
}

Matrix& BroadcastVecAdd(Matrix& Out, const Matrix& In)
{
  const OpenCLInfo &openCLInfo = Out.GetOpenCLInfo();

  uint rows  = Out.dimUInt(0) * Out.dimUInt(2) * Out.dimUInt(3);
  uint cols = Out.dimUInt(1);

  CallOpenCL("kernels/matrix_functions.cl", "gBroadcastVecAdd", openCLInfo,
      Out.data(), In.data(), rows, cols);

  return Out;
}


Matrix& BroadcastVecTanh(Matrix& Out, const Matrix& In)
{
  const OpenCLInfo &openCLInfo = Out.GetOpenCLInfo();

  uint rows  = Out.dimUInt(0) * Out.dimUInt(2) * Out.dimUInt(3);
  uint cols = Out.dimUInt(1);

  CallOpenCL("kernels/matrix_functions.cl", "gBroadcastVecTanh", openCLInfo,
      Out.data(), In.data(), rows, cols);

  return Out;
}

Matrix& BroadcastTanh(Matrix& Out, const Matrix& In, const Array<int>& batchMapping, size_t srcSize)
{
  const OpenCLInfo &openCLInfo = Out.GetOpenCLInfo();
  size_t sumOfBeamSizes = In.dim(0);

  //size_t rows = srcSize * sumOfBeamSizes;
  uint cols  = Out.dim(1);

  thread_local static Matrix Temp(openCLInfo);
  Temp.Resize(sumOfBeamSizes, cols, srcSize);

  CallOpenCL("kernels/matrix_functions.cl", "gBroadcastTanh", openCLInfo,
      Temp.data(),
      Out.data(),
      In.data(),
      (uint) srcSize,
      cols,
      batchMapping.data(),
      batchMapping.sizeUInt(),
      Temp.sizeUInt(),
      Out.sizeUInt(),
      In.sizeUInt(),
      In.dimUInt(0)
  );

  Out.Swap(Temp);
  return Out;
}

Matrix& BroadcastVecColumnAddWeighted(Matrix& Out, float weight, const Array<float>& In)
{
  const OpenCLInfo &openCLInfo = Out.GetOpenCLInfo();

  CallOpenCL("kernels/matrix_functions.cl", "gBroadcastVecColumnAddWeighted", openCLInfo,
      Out.data(),
      In.data(),
      Out.dimUInt(0),
      Out.dimUInt(1),
      weight
      );

  return Out;
}

Matrix& Slice(Matrix& Out,
              const Matrix& In,
              uint n, uint dim)
{
  Out.Resize(In.dim(0), dim);

  const OpenCLInfo &openCLInfo = Out.GetOpenCLInfo();

  CallOpenCL("kernels/matrix_functions.cl", "gSlice", openCLInfo,
      Out.data(),
      In.data(),
      n,
      dim,
      In.dimUInt(0),
      In.dimUInt(1)
      );

  return Out;
}

void PasteRows(Matrix& Out, const Matrix& In, const size_t rowNo, size_t colNo, size_t sparse)
{
  /*
  uint nColumns = In.dim(1);
  uint nRows = In.dim(0);
  cerr << "1Out=" << Out.Debug(1) << endl;
  cerr << "In=" << In.Debug(1) << endl;
  cerr << "rowNo=" << rowNo << endl;
  cerr << "colNo=" << colNo << endl;
  cerr << "sparse=" << sparse << endl;
  */
  const OpenCLInfo &openCLInfo = Out.GetOpenCLInfo();

  CallOpenCL("kernels/matrix_functions.cl", "gPasteRows", openCLInfo,
      Out.data(),
      (uint) rowNo,
      Out.dimUInt(1),
      In.data(),
      In.dimUInt(0),
      In.dimUInt(1),
      (uint) colNo,
      (uint) sparse
      );
}


void MapMatrix(Matrix& state, const Array<int>& mapping, size_t i)
{
  // blank out rows in the state matrix where the word position i does not exist
  // mapping is a concatenated array of 1 & 0 of each sentence in the batch to say whether word exists or not.

  uint batchSize = state.dim(0);
  uint sentenceLength = mapping.sizeUInt() / batchSize;

  const OpenCLInfo &openCLInfo = state.GetOpenCLInfo();

  CallOpenCL("kernels/matrix_functions.cl", "gMapMatrix", openCLInfo,
      state.data(),
      batchSize,
      state.dimUInt(1),
      sentenceLength,
      mapping.data(),
      (uint) i
      );
}

void Mean(Matrix& Out, const Matrix& In, const Array<int>& mapping)
{
  uint batchNum = Out.dimUInt(0) * Out.dimUInt(2) * Out.dimUInt(3);
  uint sentenceLength = (In.dimUInt(0) * In.dimUInt(2) * In.dimUInt(3)) / batchNum;
  uint stateLength = Out.dimUInt(1);
  //cerr << "batchNum=" << batchNum << " sentenceLength=" << sentenceLength << " stateLength=" << stateLength << endl;

  const OpenCLInfo &openCLInfo = Out.GetOpenCLInfo();

  CallOpenCL("kernels/matrix_functions.cl", "gMean", openCLInfo,
      Out.data(),
      In.data(),
      mapping.data(),
      batchNum,
      sentenceLength,
      stateLength);
}

Matrix& Softmax(Matrix& Out, const Array<int>& batchIds, const Array<int>& srcMapping,size_t srcSize)
{
  const OpenCLInfo &openCLInfo = Out.GetOpenCLInfo();

  CallOpenCL("kernels/matrix_functions.cl", "gSoftMax", openCLInfo,
      Out.data(),
      Out.dimUInt(0),
      Out.dimUInt(1),
      batchIds.data(),
      batchIds.sizeUInt(),
      srcMapping.data(),
      (uint) srcSize);

  return Out;
}

Matrix& LogSoftmax(Matrix& Out)
{
  const OpenCLInfo &openCLInfo = Out.GetOpenCLInfo();

  CallOpenCL("kernels/matrix_functions.cl", "gLogSoftMax", openCLInfo,
      Out.data(),
      Out.dimUInt(0),
      Out.dimUInt(1));

  return Out;
}

void WeightedMean(Matrix& Out,const Matrix& Weights, const Matrix& In, const Array<int>& mapping)
{
  uint numRows = Weights.dimUInt(0);
  uint numCols = In.dimUInt(1);
  uint weightsCols = Weights.dimUInt(1);
  //cerr << "WeightedMean=" << numRows << " " << numCols << " " << weightsCols << endl;

  Out.Resize(numRows, numCols);

  const OpenCLInfo &openCLInfo = Out.GetOpenCLInfo();

  CallOpenCL("kernels/matrix_functions.cl", "gWeightedMean", openCLInfo,
      Out.data(),
      Weights.data(),
      In.data(),
      mapping.data(),
      numRows,
      numCols,
      weightsCols);
}

void SetColumn(Matrix& In, int noColumn, float value)
{
  // TODO
}

void MaxElement(
    Array<float> &d_out,
    const Array<int> &d_ind,
    mblas::Matrix &d_in,
    int numBatches,
    const Array<int> &batchFirstElementIdxs)
{
  const OpenCLInfo &openCLInfo = d_out.GetOpenCLInfo();

  CallOpenCL("kernels/matrix_functions.cl", "gMaxElement", openCLInfo,
      d_out.data(),
      d_ind.data(),
      d_in.data(),
      numBatches,
      batchFirstElementIdxs.data()
      );
}

void NthElement(
    Array<float>& d_out,
    Array<unsigned> &d_ind,
    const mblas::Matrix &Probs,
    const Array<uint> &beamSizes,
    size_t maxBatchSize,
    const Array<uint> &d_cummulatedBeamSizes,
    const Array<uint> &d_batchFirstElementIdxs)
{
  const OpenCLInfo &openCLInfo = d_out.GetOpenCLInfo();

  CallOpenCL("kernels/matrix_functions.cl", "gNthElement", openCLInfo,
      Probs.data(),
      Probs.dimUInt(0),
      Probs.dimUInt(1),
      beamSizes.data(),
      beamSizes.sizeUInt(),
      d_cummulatedBeamSizes.data(),
      d_batchFirstElementIdxs.data(),
      (uint) maxBatchSize,
      d_out.data(),
      d_ind.data());
}

} // namespace mblas {
} // namespace FPGA {
} // namespace amunmt {

