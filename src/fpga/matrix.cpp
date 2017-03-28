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
,rows_(1)
,cols_(1)
{
  cl_int err;
  mem_ = clCreateBuffer(context_,  CL_MEM_READ_WRITE,  sizeof(float) * size(), NULL, &err);
  CheckError(err);
  cerr << "mem_1=" << Debug() << endl;
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
  cerr << "mem_2=" << Debug() << endl;
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
  cerr << "mem_3=" << Debug() << endl;
}

Matrix::~Matrix()
{
  Cleanup();
}

void Matrix::Cleanup()
{
  if (size()) {
    CheckError( clReleaseMemObject(mem_) );
    cerr << "Cleanup=" << mem_ << endl;
  }
}

void Matrix::Resize(size_t rows, size_t cols, size_t beam, size_t batches)
{
  Cleanup();

  rows_ = rows;
  cols_ = cols;

  cl_int err;
  mem_ = clCreateBuffer(context_,  CL_MEM_READ_WRITE,  sizeof(float) * size(), NULL, &err);
  CheckError(err);
  cerr << "mem_4=" << Debug() << endl;
}

std::string Matrix::Debug(bool detailed) const
{
  std::stringstream strm;
  strm << BaseMatrix::Debug(detailed) << " " << mem_ << " " << size();
  //cerr << "matrix=" << strm.str() << endl;

  if (detailed) {
    float sum = Sum(mem_, size(), context_, device_);
    strm << " sum=" << sum << std::flush;
  }

  return strm.str();
}

}
}
}
