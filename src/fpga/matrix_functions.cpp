#include <cassert>
#include "matrix_functions.h"
#include "matrix.h"
#include "kernel.h"
#include "array.h"

using namespace std;

namespace amunmt {
namespace FPGA {
namespace mblas {

float Sum(
    const cl_mem &mem,
    uint size,
    const OpenCLInfo &openCLInfo)
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

  global = 1024;

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
    const cl_mem &mem,
    uint size,
    const OpenCLInfo &openCLInfo)
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

  global = 1024;

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

  global = 1024;

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

  global = 1024;

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

  global = 1024;

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
                    const mblas::Matrix& Temp)
{

}


} // namespace mblas {
} // namespace FPGA {
} // namespace amunmt {

