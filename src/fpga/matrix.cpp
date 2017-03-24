#include <cassert>
#include <sstream>
#include "matrix.h"
#include "matrix_functions.h"

using namespace std;

namespace amunmt {
namespace FPGA {
namespace mblas {

Matrix::Matrix(const cl_context &context, const cl_device_id &device)
:context_(context)
,device_(device)
,rows_(0)
,cols_(0)
,mem_(nullptr)
{
}

Matrix::Matrix(const cl_context &context, const cl_device_id &device, size_t rows, size_t cols, bool zero)
:context_(context)
,device_(device)
,rows_(rows)
,cols_(cols)
{
  cl_int err;
  mem_ = clCreateBuffer(context_,  CL_MEM_READ_WRITE,  sizeof(float) * size(), NULL, &err);
  CheckError(err);
}

Matrix::Matrix(const cl_context &context, const cl_device_id &device, size_t rows, size_t cols, float *val)
:context_(context)
,device_(device)
,rows_(rows)
,cols_(cols)
{
  cl_int err;
  mem_ = clCreateBuffer(context_,  CL_MEM_COPY_HOST_PTR,  sizeof(float) * size(), val, NULL);
  CheckError(err);
}

Matrix::~Matrix()
{
  CheckError( clReleaseMemObject(mem_) );
}

void Matrix::Resize(size_t rows, size_t cols, size_t beam, size_t batches)
{
  rows_ = rows;
  cols_ = cols;

  cl_int err;
  mem_ = clCreateBuffer(context_,  CL_MEM_READ_WRITE,  sizeof(float) * size(), NULL, &err);
  CheckError(err);

}

std::string Matrix::Debug(bool detailed) const
{
  std::stringstream strm;
  strm << BaseMatrix::Debug(detailed) << " " << mem_ << " " << size();

  if (detailed) {
    float sum = Sum(mem_, size(), context_, device_);
    strm << " sum=" << sum << std::flush;
  }

  return strm.str();
}

}
}
}
