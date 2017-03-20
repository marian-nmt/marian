#include <sstream>
#include "matrix.h"

using namespace std;

namespace amunmt {
namespace FPGA {
namespace mblas {

Matrix::Matrix(const cl_context &context)
:context_(context)
,rows_(0)
,cols_(0)
{

}

Matrix::Matrix(const cl_context &context, size_t rows, size_t cols, float val)
:context_(context)
,rows_(rows)
,cols_(cols)
{

}

Matrix::Matrix(const cl_context &context, size_t rows, size_t cols, float *val)
:context_(context)
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
