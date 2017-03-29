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
  const OpenCLInfo &openCLInfo,
	Matrix& Out,
	const Matrix& In,
	const Array<uint>& indices)
{
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

  //cerr << "Out1=" << Out.Debug(true) << endl;
  //cerr << "In=" << In.Debug(true) << endl;
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
    const OpenCLInfo &openCLInfo,
		Matrix& Out,
		 const Matrix& In,
		 const Array<uint>& indices)
{
  //cerr << "indices=" << indices.Debug(true) << endl;

  Out.Resize(indices.size(), In.dim(1));
  CopyRows(openCLInfo, Out, In, indices);
  return Out;
}

void Fill(
    const OpenCLInfo &openCLInfo,
    Matrix& In,
    float value)
{
  CheckError( clEnqueueFillBuffer(openCLInfo.commands, In.data(), &value, sizeof(float), 0, In.size() * sizeof(float), 0, NULL, NULL) );
  CheckError( clFinish(openCLInfo.commands) );
}

Matrix& Prod(Matrix& C, const Matrix& A, const Matrix& B,
             bool transA, bool transB)
{

}


} // namespace mblas {
} // namespace FPGA {
} // namespace amunmt {

