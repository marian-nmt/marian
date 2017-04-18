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

float SumFloat(
    const OpenCLInfo &openCLInfo,
    const cl_mem &mem,
    uint size)
{
  cl_int err;
  size_t global;                      // global domain size for our calculation
  size_t local;                       // local domain size for our calculation

  cl_mem output = clCreateBuffer(openCLInfo.context, CL_MEM_WRITE_ONLY, sizeof(float), NULL, &err);
  CheckError(err);
  assert(output);

  // create kernel
  cl_kernel kernel = CreateKernel("kernels/matrix_functions.cl", "sum", openCLInfo);

  // Set the arguments to our compute kernel
  CheckError( clSetKernelArg(kernel, 0, sizeof(cl_mem), &mem) );
  CheckError( clSetKernelArg(kernel, 1, sizeof(cl_mem), &output) );
  CheckError( clSetKernelArg(kernel, 2, sizeof(unsigned int), &size) );

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

  // Read back the results from the device to verify the output
  //
  float results;
  CheckError( clEnqueueReadBuffer( openCLInfo.commands, output, CL_TRUE, 0, sizeof(float), &results, 0, NULL, NULL ) );

  return results;
}

unsigned int SumSizet(
    const OpenCLInfo &openCLInfo,
    const cl_mem &mem,
    uint size)
{
  cl_int err;
  size_t global;                      // global domain size for our calculation
  size_t local;                       // local domain size for our calculation

  cl_mem output = clCreateBuffer(openCLInfo.context, CL_MEM_WRITE_ONLY, sizeof(size_t), NULL, &err);
  CheckError(err);
  assert(output);

  // create kernel
  cl_kernel kernel = CreateKernel("kernels/matrix_functions.cl", "sum_size_t", openCLInfo);

  // Set the arguments to our compute kernel

  CheckError( clSetKernelArg(kernel, 0, sizeof(cl_mem), &mem) );
  CheckError( clSetKernelArg(kernel, 1, sizeof(cl_mem), &output) );
  CheckError( clSetKernelArg(kernel, 2, sizeof(uint), &size) );

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

  // Read back the results from the device to verify the output
  //
  unsigned int results;
  CheckError( clEnqueueReadBuffer( openCLInfo.commands, output, CL_TRUE, 0, sizeof(unsigned int), &results, 0, NULL, NULL ) );

  return results;
}

Matrix& Copy(Matrix& Out, const Matrix& In)
{
  const OpenCLInfo &openCLInfo = In.GetOpenCLInfo();

  Out.Resize(In.dim(0), In.dim(1), In.dim(2), In.dim(3));

  CheckError( clEnqueueCopyBuffer(openCLInfo.commands, In.data(), Out.data(), 0, 0, sizeof(float) * In.size(), 0, NULL, NULL) );

}

Matrix& CopyRows(
	Matrix& Out,
	const Matrix& In,
	const Array<uint>& indices)
{
  const OpenCLInfo &openCLInfo = In.GetOpenCLInfo();

  const cl_mem &dev = indices.data();
  size_t numPairs = indices.size();

  cl_int err;
  size_t global;                      // global domain size for our calculation
  size_t local;                       // local domain size for our calculation

  cl_mem output = clCreateBuffer(openCLInfo.context, CL_MEM_WRITE_ONLY, sizeof(float), NULL, &err);
  CheckError(err);
  assert(output);

  // create kernel
  //cerr << endl;
  //cerr << "CopyRows2=" << endl;
  cl_kernel kernel = CreateKernel("kernels/matrix_functions.cl", "gCopyRows", openCLInfo);
  //cerr << "CopyRows3=" << endl;

  // Set the arguments to our compute kernel
  size_t cols = In.dim(1);

  //cerr << "Out1=" << Out.Debug(1) << endl;
  //cerr << "In=" << In.Debug(1) << endl;
  //cerr << "cols=" << cols << endl;
  //cerr << "dev=" << indices.Debug(true) << endl;
  //cerr << "numPairs=" << numPairs << endl;

  CheckError( clSetKernelArg(kernel, 0, sizeof(cl_mem), &Out.data()) );
  CheckError( clSetKernelArg(kernel, 1, sizeof(cl_mem), &In.data()) );
  CheckError( clSetKernelArg(kernel, 2, sizeof(uint), &cols) );
  CheckError( clSetKernelArg(kernel, 3, sizeof(cl_mem), &dev) );
  CheckError( clSetKernelArg(kernel, 4, sizeof(uint), &numPairs) );

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

  //cerr << "Out2=" << Out.Debug(true) << endl;
  //cerr << "CopyRows10" << endl;

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

  cl_int err;
  size_t global;                      // global domain size for our calculation
  size_t local;                       // local domain size for our calculation

  // create kernel
  cl_kernel kernel = CreateKernel("kernels/matrix_functions.cl", "transpose", openCLInfo);

  // Set the arguments to our compute kernel
  uint rows = In.dim(0);
  uint cols = In.dim(1);

  CheckError( clSetKernelArg(kernel, 0, sizeof(cl_mem), &Out.data()) );
  CheckError( clSetKernelArg(kernel, 1, sizeof(cl_mem), &In.data()) );
  CheckError( clSetKernelArg(kernel, 2, sizeof(uint), &rows) );
  CheckError( clSetKernelArg(kernel, 3, sizeof(uint), &cols) );

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

  C.Resize(A.dim(0), B.dim(1));

  cl_int err;
  size_t global;                      // global domain size for our calculation
  size_t local;                       // local domain size for our calculation

  // create kernel
  cl_kernel kernel = CreateKernel("kernels/matrix_functions.cl", "prod", openCLInfo);

  // Set the arguments to our compute kernel
  uint rowsA = A.dim(0);
  uint colsA = A.dim(1);
  uint rowsB = B.dim(0);
  uint colsB = B.dim(1);

  CheckError( clSetKernelArg(kernel, 0, sizeof(cl_mem), &C.data()) );
  CheckError( clSetKernelArg(kernel, 1, sizeof(cl_mem), &A.data()) );
  CheckError( clSetKernelArg(kernel, 2, sizeof(cl_mem), &B.data()) );
  CheckError( clSetKernelArg(kernel, 3, sizeof(uint), &rowsA) );
  CheckError( clSetKernelArg(kernel, 4, sizeof(uint), &colsA) );
  CheckError( clSetKernelArg(kernel, 5, sizeof(uint), &rowsB) );
  CheckError( clSetKernelArg(kernel, 6, sizeof(uint), &colsB) );

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

  cl_int err;
  size_t global;                      // global domain size for our calculation
  size_t local;                       // local domain size for our calculation

  // create kernel
  cl_kernel kernel = CreateKernel("kernels/matrix_functions.cl", "gElementwiseOps", openCLInfo);

  // Set the arguments to our compute kernel
  CheckError( clSetKernelArg(kernel, 0, sizeof(cl_mem), &NextState.data()) );
  CheckError( clSetKernelArg(kernel, 1, sizeof(cl_mem), &State.data()) );
  CheckError( clSetKernelArg(kernel, 2, sizeof(cl_mem), &RUH.data()) );
  CheckError( clSetKernelArg(kernel, 3, sizeof(cl_mem), &Temp.data()) );
  CheckError( clSetKernelArg(kernel, 4, sizeof(cl_mem), &B.data()) );
  CheckError( clSetKernelArg(kernel, 5, sizeof(cl_mem), &Bx1.data()) );
  CheckError( clSetKernelArg(kernel, 6, sizeof(cl_mem), &Bx2.data()) );
  CheckError( clSetKernelArg(kernel, 7, sizeof(uint), &rows) );
  CheckError( clSetKernelArg(kernel, 8, sizeof(uint), &cols) );

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


Matrix& ElementLogit(Matrix& Out, const Matrix& In)
{
  const OpenCLInfo &openCLInfo = Out.GetOpenCLInfo();

  cl_int err;
  size_t global;                      // global domain size for our calculation
  size_t local;                       // local domain size for our calculation

  uint rows  = Out.dim(0) * Out.dim(2) * Out.dim(3);
  uint cols = Out.dim(1);

  // create kernel
  cl_kernel kernel = CreateKernel("kernels/matrix_functions.cl", "gLogit", openCLInfo);

  // Set the arguments to our compute kernel
  CheckError( clSetKernelArg(kernel, 0, sizeof(cl_mem), &Out.data()) );
  CheckError( clSetKernelArg(kernel, 1, sizeof(cl_mem), &In.data()) );
  CheckError( clSetKernelArg(kernel, 2, sizeof(uint), &rows) );
  CheckError( clSetKernelArg(kernel, 3, sizeof(uint), &cols) );


  // Get the maximum work group size for executing the kernel on the device
  //
  CheckError( clGetKernelWorkGroupInfo(kernel, openCLInfo.device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL) );

  //cerr << "CL_KERNEL_WORK_GROUP_SIZE=" << CL_KERNEL_WORK_GROUP_SIZE << endl;
  //cerr << "local=" << local << endl;

  //global = 1024;
  local = 1;
  global = 1;

  //cerr << "local=" << local << endl;
  //cerr << "global=" << global << endl;

  CheckError( clEnqueueNDRangeKernel(openCLInfo.commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL) );

  // Wait for the command commands to get serviced before reading back results
  //
  CheckError( clFinish(openCLInfo.commands) );

  return Out;
}

Matrix& ElementTanh(Matrix& Out, const Matrix& In1, const Matrix& In2)
{
  const OpenCLInfo &openCLInfo = Out.GetOpenCLInfo();

  cl_int err;
  size_t global;                      // global domain size for our calculation
  size_t local;                       // local domain size for our calculation

  uint rows  = Out.dim(0) * Out.dim(2) * Out.dim(3);
  uint cols = Out.dim(1);

  // create kernel
  cl_kernel kernel = CreateKernel("kernels/matrix_functions.cl", "gElementTanh", openCLInfo);

  // Set the arguments to our compute kernel
  CheckError( clSetKernelArg(kernel, 0, sizeof(cl_mem), &Out.data()) );
  CheckError( clSetKernelArg(kernel, 1, sizeof(cl_mem), &In1.data()) );
  CheckError( clSetKernelArg(kernel, 2, sizeof(cl_mem), &In2.data()) );
  CheckError( clSetKernelArg(kernel, 3, sizeof(uint), &rows) );
  CheckError( clSetKernelArg(kernel, 4, sizeof(uint), &cols) );


  // Get the maximum work group size for executing the kernel on the device
  //
  CheckError( clGetKernelWorkGroupInfo(kernel, openCLInfo.device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL) );

  //cerr << "CL_KERNEL_WORK_GROUP_SIZE=" << CL_KERNEL_WORK_GROUP_SIZE << endl;
  //cerr << "local=" << local << endl;

  //global = 1024;
  local = 1;
  global = 1;

  //cerr << "local=" << local << endl;
  //cerr << "global=" << global << endl;

  CheckError( clEnqueueNDRangeKernel(openCLInfo.commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL) );

  // Wait for the command commands to get serviced before reading back results
  //
  CheckError( clFinish(openCLInfo.commands) );

  return Out;
}

Matrix& ElementTanh2(Matrix& Out, const Matrix& In1, const Matrix& In2)
{
  const OpenCLInfo &openCLInfo = Out.GetOpenCLInfo();

  cl_int err;
  size_t global;                      // global domain size for our calculation
  size_t local;                       // local domain size for our calculation

  uint rows  = Out.dim(0) * Out.dim(2) * Out.dim(3);
  uint cols = Out.dim(1);

  // create kernel
  cl_kernel kernel = CreateKernel("kernels/matrix_functions.cl", "gElementTanh2", openCLInfo);

  // Set the arguments to our compute kernel
  CheckError( clSetKernelArg(kernel, 0, sizeof(cl_mem), &Out.data()) );
  CheckError( clSetKernelArg(kernel, 1, sizeof(cl_mem), &In1.data()) );
  CheckError( clSetKernelArg(kernel, 2, sizeof(cl_mem), &In2.data()) );
  CheckError( clSetKernelArg(kernel, 3, sizeof(uint), &rows) );
  CheckError( clSetKernelArg(kernel, 4, sizeof(uint), &cols) );


  // Get the maximum work group size for executing the kernel on the device
  //
  CheckError( clGetKernelWorkGroupInfo(kernel, openCLInfo.device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL) );

  //cerr << "CL_KERNEL_WORK_GROUP_SIZE=" << CL_KERNEL_WORK_GROUP_SIZE << endl;
  //cerr << "local=" << local << endl;

  //global = 1024;
  local = 1;
  global = 1;

  //cerr << "local=" << local << endl;
  //cerr << "global=" << global << endl;

  CheckError( clEnqueueNDRangeKernel(openCLInfo.commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL) );

  // Wait for the command commands to get serviced before reading back results
  //
  CheckError( clFinish(openCLInfo.commands) );

  return Out;

}

Matrix& ElementWhatever(Matrix& Out, const Matrix& In1, const Matrix& In2)
{
  const OpenCLInfo &openCLInfo = Out.GetOpenCLInfo();

  cl_int err;
  size_t global;                      // global domain size for our calculation
  size_t local;                       // local domain size for our calculation

  uint rows  = Out.dim(0) * Out.dim(2) * Out.dim(3);
  uint cols = Out.dim(1);

  // create kernel
  cl_kernel kernel = CreateKernel("kernels/matrix_functions.cl", "gElementWhatever", openCLInfo);

  // Set the arguments to our compute kernel
  CheckError( clSetKernelArg(kernel, 0, sizeof(cl_mem), &Out.data()) );
  CheckError( clSetKernelArg(kernel, 1, sizeof(cl_mem), &In1.data()) );
  CheckError( clSetKernelArg(kernel, 2, sizeof(cl_mem), &In2.data()) );
  CheckError( clSetKernelArg(kernel, 3, sizeof(uint), &rows) );
  CheckError( clSetKernelArg(kernel, 4, sizeof(uint), &cols) );


  // Get the maximum work group size for executing the kernel on the device
  //
  CheckError( clGetKernelWorkGroupInfo(kernel, openCLInfo.device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL) );

  //cerr << "CL_KERNEL_WORK_GROUP_SIZE=" << CL_KERNEL_WORK_GROUP_SIZE << endl;
  //cerr << "local=" << local << endl;

  //global = 1024;
  local = 1;
  global = 1;

  //cerr << "local=" << local << endl;
  //cerr << "global=" << global << endl;

  CheckError( clEnqueueNDRangeKernel(openCLInfo.commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL) );

  // Wait for the command commands to get serviced before reading back results
  //
  CheckError( clFinish(openCLInfo.commands) );

  return Out;
}

Matrix& ElementAddWeighted(Matrix& Out, float weight, const Matrix& In)
{
  const OpenCLInfo &openCLInfo = Out.GetOpenCLInfo();

  cl_int err;
  size_t global;                      // global domain size for our calculation
  size_t local;                       // local domain size for our calculation

  uint rows  = Out.dim(0) * Out.dim(2) * Out.dim(3);
  uint cols = Out.dim(1);

  // create kernel
  cl_kernel kernel = CreateKernel("kernels/matrix_functions.cl", "gElementAddWeighted", openCLInfo);

  // Set the arguments to our compute kernel
  CheckError( clSetKernelArg(kernel, 0, sizeof(cl_mem), &Out.data()) );
  CheckError( clSetKernelArg(kernel, 1, sizeof(cl_mem), &In.data()) );
  CheckError( clSetKernelArg(kernel, 2, sizeof(uint), &rows) );
  CheckError( clSetKernelArg(kernel, 3, sizeof(uint), &cols) );
  CheckError( clSetKernelArg(kernel, 4, sizeof(float), &weight) );


  // Get the maximum work group size for executing the kernel on the device
  //
  CheckError( clGetKernelWorkGroupInfo(kernel, openCLInfo.device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL) );

  //cerr << "CL_KERNEL_WORK_GROUP_SIZE=" << CL_KERNEL_WORK_GROUP_SIZE << endl;
  //cerr << "local=" << local << endl;

  //global = 1024;
  local = 1;
  global = 1;

  //cerr << "local=" << local << endl;
  //cerr << "global=" << global << endl;

  CheckError( clEnqueueNDRangeKernel(openCLInfo.commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL) );

  // Wait for the command commands to get serviced before reading back results
  //
  CheckError( clFinish(openCLInfo.commands) );

  return Out;
}

Matrix& BroadcastVecAdd(Matrix& Out, const Matrix& In)
{
  const OpenCLInfo &openCLInfo = Out.GetOpenCLInfo();

  cl_int err;
  size_t global;                      // global domain size for our calculation
  size_t local;                       // local domain size for our calculation

  uint rows  = Out.dim(0) * Out.dim(2) * Out.dim(3);
  uint cols = Out.dim(1);

  // create kernel
  cl_kernel kernel = CreateKernel("kernels/matrix_functions.cl", "gBroadcastVecAdd", openCLInfo);

  // Set the arguments to our compute kernel
  CheckError( clSetKernelArg(kernel, 0, sizeof(cl_mem), &Out.data()) );
  CheckError( clSetKernelArg(kernel, 1, sizeof(cl_mem), &In.data()) );
  CheckError( clSetKernelArg(kernel, 2, sizeof(uint), &rows) );
  CheckError( clSetKernelArg(kernel, 3, sizeof(uint), &cols) );


  // Get the maximum work group size for executing the kernel on the device
  //
  CheckError( clGetKernelWorkGroupInfo(kernel, openCLInfo.device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL) );

  //cerr << "CL_KERNEL_WORK_GROUP_SIZE=" << CL_KERNEL_WORK_GROUP_SIZE << endl;
  //cerr << "local=" << local << endl;

  //global = 1024;
  local = 1;
  global = 1;

  //cerr << "local=" << local << endl;
  //cerr << "global=" << global << endl;

  CheckError( clEnqueueNDRangeKernel(openCLInfo.commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL) );

  // Wait for the command commands to get serviced before reading back results
  //
  CheckError( clFinish(openCLInfo.commands) );

  return Out;
}


Matrix& BroadcastVecTanh(Matrix& Out, const Matrix& In)
{
  const OpenCLInfo &openCLInfo = Out.GetOpenCLInfo();

  cl_int err;
  size_t global;                      // global domain size for our calculation
  size_t local;                       // local domain size for our calculation

  uint rows  = Out.dim(0) * Out.dim(2) * Out.dim(3);
  uint cols = Out.dim(1);

  // create kernel
  cl_kernel kernel = CreateKernel("kernels/matrix_functions.cl", "gBroadcastVecTanh", openCLInfo);

  // Set the arguments to our compute kernel
  CheckError( clSetKernelArg(kernel, 0, sizeof(cl_mem), &Out.data()) );
  CheckError( clSetKernelArg(kernel, 1, sizeof(cl_mem), &In.data()) );
  CheckError( clSetKernelArg(kernel, 2, sizeof(uint), &rows) );
  CheckError( clSetKernelArg(kernel, 3, sizeof(uint), &cols) );


  // Get the maximum work group size for executing the kernel on the device
  //
  CheckError( clGetKernelWorkGroupInfo(kernel, openCLInfo.device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL) );

  //cerr << "CL_KERNEL_WORK_GROUP_SIZE=" << CL_KERNEL_WORK_GROUP_SIZE << endl;
  //cerr << "local=" << local << endl;

  //global = 1024;
  local = 1;
  global = 1;

  //cerr << "local=" << local << endl;
  //cerr << "global=" << global << endl;

  CheckError( clEnqueueNDRangeKernel(openCLInfo.commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL) );

  // Wait for the command commands to get serviced before reading back results
  //
  CheckError( clFinish(openCLInfo.commands) );

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

  cl_int err;
  size_t global;                      // global domain size for our calculation
  size_t local;                       // local domain size for our calculation

  // create kernel
  cl_kernel kernel = CreateKernel("kernels/matrix_functions.cl", "gBroadcastTanh", openCLInfo);

  // Set the arguments to our compute kernel
  uint srcSizeUint = srcSize;
  uint batchSize = batchMapping.size();
  uint tempSize = Temp.size();
  uint outSize = Out.size();
  uint inSize = In.size();
  uint inRows = In.dim(0);

  CheckError( clSetKernelArg(kernel, 0, sizeof(cl_mem), &Temp.data()) );
  CheckError( clSetKernelArg(kernel, 1, sizeof(cl_mem), &Out.data()) );
  CheckError( clSetKernelArg(kernel, 2, sizeof(cl_mem), &In.data()) );
  CheckError( clSetKernelArg(kernel, 3, sizeof(uint), &srcSizeUint) );
  CheckError( clSetKernelArg(kernel, 4, sizeof(uint), &batchSize) );
  CheckError( clSetKernelArg(kernel, 5, sizeof(uint), &cols) );
  CheckError( clSetKernelArg(kernel, 6, sizeof(cl_mem), &batchMapping.data()) );
  CheckError( clSetKernelArg(kernel, 7, sizeof(uint), &batchSize) );
  CheckError( clSetKernelArg(kernel, 8, sizeof(uint), &tempSize) );
  CheckError( clSetKernelArg(kernel, 9, sizeof(uint), &outSize) );
  CheckError( clSetKernelArg(kernel, 10, sizeof(uint), &inSize) );
  CheckError( clSetKernelArg(kernel, 11, sizeof(uint), &inRows) );

  // Get the maximum work group size for executing the kernel on the device
  //
  CheckError( clGetKernelWorkGroupInfo(kernel, openCLInfo.device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL) );

  //cerr << "CL_KERNEL_WORK_GROUP_SIZE=" << CL_KERNEL_WORK_GROUP_SIZE << endl;
  //cerr << "local=" << local << endl;

  //global = 1024;
  local = 1;
  global = 1;

  //cerr << "local=" << local << endl;
  //cerr << "global=" << global << endl;

  CheckError( clEnqueueNDRangeKernel(openCLInfo.commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL) );

  // Wait for the command commands to get serviced before reading back results
  //
  CheckError( clFinish(openCLInfo.commands) );

  Out.Swap(Temp);
  return Out;
}

Matrix& BroadcastVecColumnAddWeighted(Matrix& Out, float weight, const Array<float>& In)
{
  const OpenCLInfo &openCLInfo = Out.GetOpenCLInfo();

  cl_int err;
  size_t global;                      // global domain size for our calculation
  size_t local;                       // local domain size for our calculation

  // create kernel
  cl_kernel kernel = CreateKernel("kernels/matrix_functions.cl", "gBroadcastVecColumnAddWeighted", openCLInfo);

  // Set the arguments to our compute kernel
  uint rows = Out.dim(0);
  uint cols = Out.dim(1);

  CheckError( clSetKernelArg(kernel, 0, sizeof(cl_mem), &Out.data()) );
  CheckError( clSetKernelArg(kernel, 1, sizeof(cl_mem), &In.data()) );
  CheckError( clSetKernelArg(kernel, 2, sizeof(uint), &rows) );
  CheckError( clSetKernelArg(kernel, 3, sizeof(uint), &cols) );
  CheckError( clSetKernelArg(kernel, 4, sizeof(float), &weight) );

  // Get the maximum work group size for executing the kernel on the device
  //
  CheckError( clGetKernelWorkGroupInfo(kernel, openCLInfo.device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL) );

  //cerr << "CL_KERNEL_WORK_GROUP_SIZE=" << CL_KERNEL_WORK_GROUP_SIZE << endl;
  //cerr << "local=" << local << endl;

  //global = 1024;
  local = 1;
  global = 1;

  //cerr << "local=" << local << endl;
  //cerr << "global=" << global << endl;

  CheckError( clEnqueueNDRangeKernel(openCLInfo.commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL) );

  // Wait for the command commands to get serviced before reading back results
  //
  CheckError( clFinish(openCLInfo.commands) );

  return Out;

}

Matrix& Slice(Matrix& Out,
              const Matrix& In,
              uint n, uint dim)
{
  Out.Resize(In.dim(0), dim);

  const OpenCLInfo &openCLInfo = Out.GetOpenCLInfo();

  cl_int err;
  size_t global;                      // global domain size for our calculation
  size_t local;                       // local domain size for our calculation

  // create kernel
  cl_kernel kernel = CreateKernel("kernels/matrix_functions.cl", "gSlice", openCLInfo);

  // Set the arguments to our compute kernel
  uint rows = In.dim(0);
  uint cols = In.dim(1);

  CheckError( clSetKernelArg(kernel, 0, sizeof(cl_mem), &Out.data()) );
  CheckError( clSetKernelArg(kernel, 1, sizeof(cl_mem), &In.data()) );
  CheckError( clSetKernelArg(kernel, 2, sizeof(uint), &n) );
  CheckError( clSetKernelArg(kernel, 3, sizeof(uint), &dim) );
  CheckError( clSetKernelArg(kernel, 4, sizeof(uint), &rows) );
  CheckError( clSetKernelArg(kernel, 5, sizeof(uint), &cols) );

  // Get the maximum work group size for executing the kernel on the device
  //
  CheckError( clGetKernelWorkGroupInfo(kernel, openCLInfo.device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL) );

  //cerr << "CL_KERNEL_WORK_GROUP_SIZE=" << CL_KERNEL_WORK_GROUP_SIZE << endl;
  //cerr << "local=" << local << endl;

  //global = 1024;
  local = 1;
  global = 1;

  //cerr << "local=" << local << endl;
  //cerr << "global=" << global << endl;

  CheckError( clEnqueueNDRangeKernel(openCLInfo.commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL) );

  // Wait for the command commands to get serviced before reading back results
  //
  CheckError( clFinish(openCLInfo.commands) );

  return Out;
}

void PasteRows(Matrix& Out, const Matrix& In, const size_t rowNo, size_t colNo, size_t sparse)
{
  uint nColumns = In.dim(1);
  uint nRows = In.dim(0);
  /*
  cerr << "1Out=" << Out.Debug(1) << endl;
  cerr << "In=" << In.Debug(1) << endl;
  cerr << "rowNo=" << rowNo << endl;
  cerr << "colNo=" << colNo << endl;
  cerr << "sparse=" << sparse << endl;
  */
  const OpenCLInfo &openCLInfo = Out.GetOpenCLInfo();

  cl_int err;
  size_t global;                      // global domain size for our calculation
  size_t local;                       // local domain size for our calculation

  // create kernel
  cl_kernel kernel = CreateKernel("kernels/matrix_functions.cl", "gPasteRows", openCLInfo);

  // Set the arguments to our compute kernel
  uint outCols = Out.dim(1);
  uint inRows = In.dim(0);
  uint inCols = In.dim(1);
  uint sparseUint = sparse;

  CheckError( clSetKernelArg(kernel, 0, sizeof(cl_mem), &Out.data()) );
  CheckError( clSetKernelArg(kernel, 1, sizeof(uint), &rowNo) );
  CheckError( clSetKernelArg(kernel, 2, sizeof(uint), &outCols) );

  CheckError( clSetKernelArg(kernel, 3, sizeof(cl_mem), &In.data()) );
  CheckError( clSetKernelArg(kernel, 4, sizeof(uint), &inRows) );
  CheckError( clSetKernelArg(kernel, 5, sizeof(uint), &inCols) );

  CheckError( clSetKernelArg(kernel, 6, sizeof(uint), &colNo) );
  CheckError( clSetKernelArg(kernel, 7, sizeof(uint), &sparseUint) );

  // Get the maximum work group size for executing the kernel on the device
  //
  CheckError( clGetKernelWorkGroupInfo(kernel, openCLInfo.device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL) );

  //cerr << "CL_KERNEL_WORK_GROUP_SIZE=" << CL_KERNEL_WORK_GROUP_SIZE << endl;
  //cerr << "local=" << local << endl;

  //global = 1024;
  local = 1;
  global = 1;

  //cerr << "local=" << local << endl;
  //cerr << "global=" << global << endl;

  CheckError( clEnqueueNDRangeKernel(openCLInfo.commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL) );

  // Wait for the command commands to get serviced before reading back results
  //
  CheckError( clFinish(openCLInfo.commands) );

  //cerr << "2Out=" << Out.Debug(1) << endl;

}


void MapMatrix(Matrix& state, const Array<int>& mapping, size_t i)
{
  int batchSize = state.dim(0);
  int stateLength = state.dim(1);
  int sentenceLength = mapping.size() / batchSize;

  // TODO
  //cerr << "MapMatrix=" << state.Debug(1) << endl;

}

void Mean(Matrix& Out, const Matrix& In, const Array<int>& mapping)
{
  uint batchNum = Out.dim(0) * Out.dim(2) * Out.dim(3);
  uint sentenceLength = (In.dim(0) * In.dim(2) * In.dim(3)) / batchNum;
  uint stateLength = Out.dim(1);
  //cerr << "batchNum=" << batchNum << " sentenceLength=" << sentenceLength << " stateLength=" << stateLength << endl;

  const OpenCLInfo &openCLInfo = Out.GetOpenCLInfo();

  cl_int err;
  size_t global;                      // global domain size for our calculation
  size_t local;                       // local domain size for our calculation

  // create kernel
  cl_kernel kernel = CreateKernel("kernels/matrix_functions.cl", "gMean", openCLInfo);

  // Set the arguments to our compute kernel

  CheckError( clSetKernelArg(kernel, 0, sizeof(cl_mem), &Out.data()) );
  CheckError( clSetKernelArg(kernel, 1, sizeof(cl_mem), &In.data()) );
  CheckError( clSetKernelArg(kernel, 2, sizeof(cl_mem), &mapping.data()) );
  CheckError( clSetKernelArg(kernel, 3, sizeof(uint), &batchNum) );
  CheckError( clSetKernelArg(kernel, 4, sizeof(uint), &sentenceLength) );
  CheckError( clSetKernelArg(kernel, 5, sizeof(uint), &stateLength) );

  // Get the maximum work group size for executing the kernel on the device
  //
  CheckError( clGetKernelWorkGroupInfo(kernel, openCLInfo.device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL) );

  //cerr << "CL_KERNEL_WORK_GROUP_SIZE=" << CL_KERNEL_WORK_GROUP_SIZE << endl;
  //cerr << "local=" << local << endl;

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

Matrix& Softmax(Matrix& Out, const Array<int>& batchIds, const Array<int>& srcMapping,size_t srcSize)
{
  const OpenCLInfo &openCLInfo = Out.GetOpenCLInfo();

  cl_int err;
  size_t global;                      // global domain size for our calculation
  size_t local;                       // local domain size for our calculation

  // create kernel
  cl_kernel kernel = CreateKernel("kernels/matrix_functions.cl", "gSoftMax", openCLInfo);

  // Set the arguments to our compute kernel
  uint outRows = Out.dim(0);
  uint outCols = Out.dim(1);
  uint batchIdsSize = batchIds.size();
  uint srcSizeUint = srcSize;

  CheckError( clSetKernelArg(kernel, 0, sizeof(cl_mem), &Out.data()) );
  CheckError( clSetKernelArg(kernel, 1, sizeof(uint), &outRows) );
  CheckError( clSetKernelArg(kernel, 2, sizeof(uint), &outCols) );
  CheckError( clSetKernelArg(kernel, 3, sizeof(cl_mem), &batchIds.data()) );
  CheckError( clSetKernelArg(kernel, 4, sizeof(uint), &batchIdsSize) );
  CheckError( clSetKernelArg(kernel, 5, sizeof(cl_mem), &srcMapping.data()) );
  CheckError( clSetKernelArg(kernel, 6, sizeof(uint), &srcSizeUint) );

  // Get the maximum work group size for executing the kernel on the device
  //
  CheckError( clGetKernelWorkGroupInfo(kernel, openCLInfo.device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL) );

  //cerr << "CL_KERNEL_WORK_GROUP_SIZE=" << CL_KERNEL_WORK_GROUP_SIZE << endl;
  //cerr << "local=" << local << endl;

  //global = 1024;
  local = 1;
  global = 1;

  //cerr << "local=" << local << endl;
  //cerr << "global=" << global << endl;

  CheckError( clEnqueueNDRangeKernel(openCLInfo.commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL) );

  // Wait for the command commands to get serviced before reading back results
  //
  CheckError( clFinish(openCLInfo.commands) );

  return Out;
}

Matrix& LogSoftmax(Matrix& Out)
{
  const OpenCLInfo &openCLInfo = Out.GetOpenCLInfo();

  cl_int err;
  size_t global;                      // global domain size for our calculation
  size_t local;                       // local domain size for our calculation

  // create kernel
  cl_kernel kernel = CreateKernel("kernels/matrix_functions.cl", "gLogSoftMax", openCLInfo);

  // Set the arguments to our compute kernel
  uint outRows = Out.dim(0);
  uint outCols = Out.dim(1);

  CheckError( clSetKernelArg(kernel, 0, sizeof(cl_mem), &Out.data()) );
  CheckError( clSetKernelArg(kernel, 1, sizeof(uint), &outRows) );
  CheckError( clSetKernelArg(kernel, 2, sizeof(uint), &outCols) );

  // Get the maximum work group size for executing the kernel on the device
  //
  CheckError( clGetKernelWorkGroupInfo(kernel, openCLInfo.device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL) );

  //cerr << "CL_KERNEL_WORK_GROUP_SIZE=" << CL_KERNEL_WORK_GROUP_SIZE << endl;
  //cerr << "local=" << local << endl;

  //global = 1024;
  local = 1;
  global = 1;

  //cerr << "local=" << local << endl;
  //cerr << "global=" << global << endl;

  CheckError( clEnqueueNDRangeKernel(openCLInfo.commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL) );

  // Wait for the command commands to get serviced before reading back results
  //
  CheckError( clFinish(openCLInfo.commands) );

  return Out;
}

void WeightedMean(Matrix& Out,const Matrix& Weights, const Matrix& In, const Array<int>& mapping)
{
  uint numRows = Weights.dim(0);
  uint numCols = In.dim(1);
  uint weightsCols = Weights.dim(1);

  Out.Resize(numRows, numCols);

  const OpenCLInfo &openCLInfo = Out.GetOpenCLInfo();

  cl_int err;
  size_t global;                      // global domain size for our calculation
  size_t local;                       // local domain size for our calculation

  // create kernel
  cl_kernel kernel = CreateKernel("kernels/matrix_functions.cl", "gWeightedMean", openCLInfo);

  // Set the arguments to our compute kernel
  CheckError( clSetKernelArg(kernel, 0, sizeof(cl_mem), &Out.data()) );
  CheckError( clSetKernelArg(kernel, 1, sizeof(cl_mem), &Weights.data()) );
  CheckError( clSetKernelArg(kernel, 2, sizeof(cl_mem), &In.data()) );
  CheckError( clSetKernelArg(kernel, 3, sizeof(cl_mem), &mapping.data()) );
  CheckError( clSetKernelArg(kernel, 4, sizeof(uint), &numRows) );
  CheckError( clSetKernelArg(kernel, 5, sizeof(uint), &numCols) );
  CheckError( clSetKernelArg(kernel, 6, sizeof(uint), &weightsCols) );

  // Get the maximum work group size for executing the kernel on the device
  //
  CheckError( clGetKernelWorkGroupInfo(kernel, openCLInfo.device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL) );

  //cerr << "CL_KERNEL_WORK_GROUP_SIZE=" << CL_KERNEL_WORK_GROUP_SIZE << endl;
  //cerr << "local=" << local << endl;

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

  cl_int err;
  size_t global;                      // global domain size for our calculation
  size_t local;                       // local domain size for our calculation

  // create kernel
  cl_kernel kernel = CreateKernel("kernels/matrix_functions.cl", "gMaxElement", openCLInfo);

  // Set the arguments to our compute kernel
  CheckError( clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_out.data()) );
  CheckError( clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_ind.data()) );
  CheckError( clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_in.data()) );
  CheckError( clSetKernelArg(kernel, 3, sizeof(int), &numBatches) );
  CheckError( clSetKernelArg(kernel, 4, sizeof(cl_mem), &batchFirstElementIdxs.data()) );

  // Get the maximum work group size for executing the kernel on the device
  //
  CheckError( clGetKernelWorkGroupInfo(kernel, openCLInfo.device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL) );

  //cerr << "CL_KERNEL_WORK_GROUP_SIZE=" << CL_KERNEL_WORK_GROUP_SIZE << endl;
  //cerr << "local=" << local << endl;

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

void NthElement(
    Array<float>& d_out,
    Array<int> &d_ind,
    const mblas::Matrix &Probs,
    size_t maxBeamSize,
    size_t maxBatchSize)
{
  const OpenCLInfo &openCLInfo = d_out.GetOpenCLInfo();

  cl_int err;
  size_t global;                      // global domain size for our calculation
  size_t local;                       // local domain size for our calculation

  // create kernel
  cl_kernel kernel = CreateKernel("kernels/matrix_functions.cl", "gNthElement", openCLInfo);

  // Set the arguments to our compute kernel
  uint ProbsRows = Probs.dim(0);
  uint ProbsCols = Probs.dim(1);
  uint maxBeamSizeUint = maxBeamSize;
  uint maxBatchSizeUint = maxBatchSize;

  CheckError( clSetKernelArg(kernel, 0, sizeof(cl_mem), &Probs.data()) );
  CheckError( clSetKernelArg(kernel, 1, sizeof(uint), &ProbsRows) );
  CheckError( clSetKernelArg(kernel, 2, sizeof(uint), &ProbsCols) );
  CheckError( clSetKernelArg(kernel, 3, sizeof(uint), &maxBeamSizeUint) );
  CheckError( clSetKernelArg(kernel, 4, sizeof(uint), &maxBatchSizeUint) );
  CheckError( clSetKernelArg(kernel, 5, sizeof(cl_mem), &d_out.data()) );
  CheckError( clSetKernelArg(kernel, 6, sizeof(cl_mem), &d_ind.data()) );

  // Get the maximum work group size for executing the kernel on the device
  //
  CheckError( clGetKernelWorkGroupInfo(kernel, openCLInfo.device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL) );

  //cerr << "CL_KERNEL_WORK_GROUP_SIZE=" << CL_KERNEL_WORK_GROUP_SIZE << endl;
  //cerr << "local=" << local << endl;

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

} // namespace mblas {
} // namespace FPGA {
} // namespace amunmt {

