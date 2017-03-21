#include <cassert>
#include <sstream>
#include "matrix.h"

using namespace std;

namespace amunmt {
namespace FPGA {
namespace mblas {

Matrix::Matrix(const cl_context &context, const cl_device_id &device)
:context_(context)
,device_(device)
,rows_(0)
,cols_(0)
{

}

Matrix::Matrix(const cl_context &context, const cl_device_id &device, size_t rows, size_t cols, float val)
:context_(context)
,device_(device)
,rows_(rows)
,cols_(cols)
{

}

Matrix::Matrix(const cl_context &context, const cl_device_id &device, size_t rows, size_t cols, float *val)
:context_(context)
,device_(device)
,rows_(rows)
,cols_(cols)
{
  mem_ = clCreateBuffer(context_,  CL_MEM_COPY_HOST_PTR,  sizeof(float) * rows * cols, val, NULL);
}

void Matrix::Resize(size_t rows, size_t cols, size_t beam, size_t batches)
{
  rows_ = rows;
  cols_ = cols;

  //clReleaseMemObject(mem_);
  mem_ = clCreateBuffer(context_,  CL_MEM_READ_WRITE,  sizeof(float) * rows * cols, NULL, NULL);

}

float Matrix::Sum()
{
  int err;

 cl_mem output = clCreateBuffer(context_, CL_MEM_WRITE_ONLY, sizeof(float), NULL, &err);
  CheckError(err);
  assert(output);
/*
  // create kernel
  cl_command_queue commands = CreateCommandQueue(context_, devices_[0]);
  cl_kernel kernel = CreateKernel("kernels/sum.cl", context_, devices_[0]);

  // Set the arguments to our compute kernel
  unsigned int count = size();

  CheckError( clSetKernelArg(kernel, 0, sizeof(cl_mem), &mem_) );
  CheckError( clSetKernelArg(kernel, 1, sizeof(cl_mem), &output) );
  CheckError( clSetKernelArg(kernel, 2, sizeof(unsigned int), &count) );
*/
}

std::string Matrix::Debug(bool detailed) const
{
  std::stringstream strm;
  strm << BaseMatrix::Debug(detailed) << " ";

  if (detailed) {
    //float sum = Sum(data(), size());
    //strm << "size=" << size() << " sum=" << sum << std::flush;
  }

  return strm.str();
}

}
}
}
