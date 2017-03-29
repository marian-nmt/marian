#include <cassert>
#include <sstream>
#include "matrix.h"
#include "matrix_functions.h"
#include "types-fpga.h"
#include "kernel.h"

using namespace std;

namespace amunmt {
namespace FPGA {
namespace mblas {

Matrix::Matrix(const OpenCLInfo &openCLInfo)
:openCLInfo_(openCLInfo)
,rows_(0)
,cols_(0)
,mem_(nullptr)
{
  /*
  cl_int err;
  mem_ = clCreateBuffer(context_,  CL_MEM_READ_WRITE,  sizeof(float) * size(), NULL, &err);
  CheckError(err);
  cerr << "mem_1=" << Debug() << endl;
  */
  //cerr << "mem_1=" << Debug() << endl;

}

Matrix::Matrix(const OpenCLInfo &openCLInfo, size_t rows, size_t cols, bool zero)
:openCLInfo_(openCLInfo)
,rows_(rows)
,cols_(cols)
{
  cl_int err;
  mem_ = clCreateBuffer(openCLInfo_.context,  CL_MEM_READ_WRITE,  sizeof(float) * size(), NULL, &err);
  CheckError(err);
  //cerr << "mem_2=" << Debug() << endl;
}

Matrix::Matrix(const OpenCLInfo &openCLInfo, size_t rows, size_t cols, float *val)
:openCLInfo_(openCLInfo)
,rows_(rows)
,cols_(cols)
{
  cl_int err;
  mem_ = clCreateBuffer(openCLInfo_.context,  CL_MEM_COPY_HOST_PTR,  sizeof(float) * size(), val, NULL);
  CheckError(err);
  //cerr << "mem_3=" << Debug() << " " << *val << endl;
}

Matrix::Matrix(const Matrix &other)
:Matrix(other.openCLInfo_, other.rows_, other.cols_)
{
  cl_command_queue commands = CreateCommandQueue(openCLInfo_);
  CheckError( clEnqueueCopyBuffer(commands, other.data(), data(), 0, 0, sizeof(float) * size(), 0, NULL, NULL) );
}

Matrix::~Matrix()
{
  Cleanup();
}

void Matrix::Cleanup()
{
  if (size()) {
    //CheckError( clReleaseMemObject(mem_) );
    //cerr << "Cleanup=" << this << " " << mem_ << endl;
  }
}

void Matrix::Resize(size_t rows, size_t cols, size_t beam, size_t batches)
{
  //cerr << "resize" << endl;
  Cleanup();

  rows_ = rows;
  cols_ = cols;

  cl_int err;
  mem_ = clCreateBuffer(openCLInfo_.context,  CL_MEM_READ_WRITE,  sizeof(float) * size(), NULL, &err);
  CheckError(err);
  //cerr << "mem_4=" << Debug() << endl;
}

std::string Matrix::Debug(bool detailed) const
{
  std::stringstream strm;
  strm << BaseMatrix::Debug(detailed) << " " << mem_;
  //cerr << "Debug1=" << strm.str() << endl;

  if (detailed) {
    //cerr << "Debug2" << endl;
    float sum = Sum(mem_, size(), openCLInfo_);
    //cerr << "Debug3" << endl;
    strm << " sum=" << sum << std::flush;
    //cerr << "Debug4" << endl;
  }
  //cerr << "Debug5" << endl;

  return strm.str();
}

}
}
}
