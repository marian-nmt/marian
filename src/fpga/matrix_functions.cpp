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
    const cl_context &context,
    const cl_device_id &device)
{
  cl_int err;
  size_t global;                      // global domain size for our calculation
  size_t local;                       // local domain size for our calculation

  cl_mem output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float), NULL, &err);
  CheckError(err);
  assert(output);

  // create kernel
  cl_command_queue commands = CreateCommandQueue(context, device);
  cl_kernel kernel = CreateKernel("kernels/matrix_functions.cl", "sum", context, device);

  // Set the arguments to our compute kernel
  CheckError( clSetKernelArg(kernel, 0, sizeof(cl_mem), &mem) );
  CheckError( clSetKernelArg(kernel, 1, sizeof(cl_mem), &output) );
  CheckError( clSetKernelArg(kernel, 2, sizeof(unsigned int), &size) );

  // Get the maximum work group size for executing the kernel on the device
  //
  CheckError( clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL) );

  global = 1024;

  //cerr << "local=" << local << endl;
  //cerr << "global=" << global << endl;

  CheckError( clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL) );

  // Wait for the command commands to get serviced before reading back results
  //
  CheckError( clFinish(commands) );

  // Read back the results from the device to verify the output
  //
  float results;
  CheckError( clEnqueueReadBuffer( commands, output, CL_TRUE, 0, sizeof(float), &results, 0, NULL, NULL ) );

  return results;
}

unsigned int SumSizet(
    const cl_mem &mem,
    uint size,
    const cl_context &context,
    const cl_device_id &device)
{
  cl_int err;
  size_t global;                      // global domain size for our calculation
  size_t local;                       // local domain size for our calculation

  cl_mem output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(size_t), NULL, &err);
  CheckError(err);
  assert(output);

  // create kernel
  cl_command_queue commands = CreateCommandQueue(context, device);
  cl_kernel kernel = CreateKernel("kernels/matrix_functions.cl", "sum_size_t", context, device);

  // Set the arguments to our compute kernel

  CheckError( clSetKernelArg(kernel, 0, sizeof(cl_mem), &mem) );
  CheckError( clSetKernelArg(kernel, 1, sizeof(cl_mem), &output) );
  CheckError( clSetKernelArg(kernel, 2, sizeof(uint), &size) );

  // Get the maximum work group size for executing the kernel on the device
  //
  CheckError( clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL) );

  global = 1024;

  //cerr << "local=" << local << endl;
  //cerr << "global=" << global << endl;

  CheckError( clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL) );

  // Wait for the command commands to get serviced before reading back results
  //
  CheckError( clFinish(commands) );

  // Read back the results from the device to verify the output
  //
  unsigned int results;
  CheckError( clEnqueueReadBuffer( commands, output, CL_TRUE, 0, sizeof(unsigned int), &results, 0, NULL, NULL ) );

  return results;
}

Matrix& CopyRows(
	const cl_context &context,
	const cl_device_id &device,
	Matrix& Out,
	const Matrix& In,
	const Array<unsigned int>& indices)
{
  const cl_mem &dev = indices.data();
  size_t numPairs = indices.size();

  cl_int err;
  size_t global;                      // global domain size for our calculation
  size_t local;                       // local domain size for our calculation

  cl_mem output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float), NULL, &err);
  CheckError(err);
  assert(output);

  // create kernel
  cerr << "CopyRows1=" << endl;
  cl_command_queue commands = CreateCommandQueue(context, device);
  cerr << "CopyRows2=" << endl;
  cl_kernel kernel = CreateKernel("kernels/matrix_functions.cl", "gCopyRows", context, device);
  cerr << "CopyRows3=" << endl;

  // Set the arguments to our compute kernel
  size_t cols = In.dim(1);

  cerr << "Out1=" << Out.Debug() << endl;
  cerr << "In=" << In.Debug() << endl;
  cerr << "cols=" << cols << endl;
  cerr << "dev=" << indices.Debug(true) << endl;
  cerr << "numPairs=" << numPairs << endl;

  CheckError( clSetKernelArg(kernel, 0, sizeof(cl_mem), &Out.data()) );
  CheckError( clSetKernelArg(kernel, 1, sizeof(cl_mem), &In.data()) );
  CheckError( clSetKernelArg(kernel, 2, sizeof(unsigned int), &cols) );
  CheckError( clSetKernelArg(kernel, 3, sizeof(cl_mem), &dev) );
  CheckError( clSetKernelArg(kernel, 4, sizeof(unsigned int), &numPairs) );

  // Get the maximum work group size for executing the kernel on the device
  //
  CheckError( clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL) );

  global = 1024;

  //cerr << "local=" << local << endl;
  //cerr << "global=" << global << endl;

  CheckError( clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL) );

  // Wait for the command commands to get serviced before reading back results
  //
  CheckError( clFinish(commands) );

  cerr << "Out2=" << Out.Debug() << endl;
  cerr << "CopyRows10" << endl;

  return Out;

}

Matrix& Assemble(
		const cl_context &context,
		const cl_device_id &device,
		Matrix& Out,
		 const Matrix& In,
		 const Array<unsigned int>& indices)
{
  cerr << "indices=" << indices.Debug(true) << endl;

  Out.Resize(indices.size(), In.dim(1));
  CopyRows(context, device, Out, In, indices);
  return Out;
}

}
}
}

