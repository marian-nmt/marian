#include <cassert>
#include "matrix_functions.h"
#include "matrix.h"
#include "kernel.h"

namespace amunmt {
namespace FPGA {
namespace mblas {

float Sum(
    const cl_mem &mem,
    size_t size,
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
  cl_kernel kernel = CreateKernel("kernels/sum.cl", "sum", context, device);

  // Set the arguments to our compute kernel
  unsigned int count = size;

  CheckError( clSetKernelArg(kernel, 0, sizeof(cl_mem), &mem) );
  CheckError( clSetKernelArg(kernel, 1, sizeof(cl_mem), &output) );
  CheckError( clSetKernelArg(kernel, 2, sizeof(unsigned int), &count) );

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

Matrix& CopyRows(Matrix& Out,
                 const Matrix& In,
                 const cl_mem &dev,
                 size_t numPairs)
{

}

Matrix& Assemble(Matrix& Out,
                 const Matrix& In,
                 const Array<size_t>& indeces)
{
  Out.Resize(indeces.size(), In.dim(1));
  CopyRows(Out, In, indeces.data(), indeces.size());
  return Out;
}

}
}
}

