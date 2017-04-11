#include <cassert>
#include <sstream>
#include "matrix.h"
#include "matrix_functions.h"
#include "types-fpga.h"
#include "kernel.h"
#include "common/exception.h"

using namespace std;

namespace amunmt {
namespace FPGA {
namespace mblas {

Matrix::Matrix(const OpenCLInfo &openCLInfo)
:openCLInfo_(openCLInfo)
,rows_(0)
,cols_(0)
,beam_(0)
,batches_(0)
,arrSize_(0)
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
,beam_(1)
,batches_(1)
,arrSize_(size())
{
  cl_int err;
  mem_ = clCreateBuffer(openCLInfo_.context,  CL_MEM_READ_WRITE,  sizeof(float) * size(), NULL, &err);
  CheckError(err);
  //cerr << "mem_2=" << Debug() << endl;

  if (zero) {
    Fill(*this, 0);
  }

}

Matrix::Matrix(const OpenCLInfo &openCLInfo, size_t rows, size_t cols, float *val)
:openCLInfo_(openCLInfo)
,rows_(rows)
,cols_(cols)
,beam_(1)
,batches_(1)
,arrSize_(size())
{
  cl_int err;
  mem_ = clCreateBuffer(openCLInfo_.context,  CL_MEM_COPY_HOST_PTR,  sizeof(float) * size(), val, NULL);
  CheckError(err);
  //cerr << "mem_3=" << Debug() << " " << *val << endl;
}

Matrix::Matrix(const Matrix &other)
:Matrix(other.openCLInfo_, other.rows_, other.cols_)
{
  CheckError( clEnqueueCopyBuffer(openCLInfo_.commands, other.data(), data(), 0, 0, sizeof(float) * size(), 0, NULL, NULL) );
}

Matrix::Matrix(Matrix &&other)
:openCLInfo_(other.openCLInfo_)
,mem_(other.mem_)
,rows_(other.rows_)
,cols_(other.cols_)
,beam_(other.beam_)
,batches_(other.batches_)
,arrSize_(other.arrSize_)
{
  other.mem_ = nullptr;
  other.rows_ = 0;
  other.cols_ = 0;
  other.beam_ = 0;
  other.batches_ = 0;
  other.arrSize_ = 0;
}


Matrix::~Matrix()
{
}

void Matrix::Resize(size_t rows, size_t cols, size_t beam, size_t batches)
{
  cl_int err;
  size_t newSize = cols * rows * beam * batches;
  if (newSize > arrSize_) {
    //cerr << "resize: clCreateBuffer " << newSize << endl;
    cl_mem newMem = clCreateBuffer(openCLInfo_.context,  CL_MEM_READ_WRITE,  sizeof(float) * newSize, NULL, &err);
    CheckError(err);

    size_t oldSize = size();
    assert(newSize > oldSize);

    if (oldSize) {
      //cerr << "resize: clEnqueueCopyBuffer " << oldSize << endl;
      CheckError( clEnqueueCopyBuffer(openCLInfo_.commands, mem_, newMem, 0, 0, sizeof(float) * oldSize, 0, NULL, NULL) );
    }

    mem_ = newMem;
    arrSize_ = newSize;
  }

  rows_ = rows;
  cols_ = cols;
  beam_ = beam;
  batches_ = batches;
}

void Matrix::Reshape(size_t rows, size_t cols, size_t beam, size_t batches)
{
  size_t newSize = cols * rows * beam * batches;
  amunmt_UTIL_THROW_IF2(newSize > arrSize_, "Must reshape to same or smaller size");

  rows_ = rows;
  cols_ = cols;
  beam_ = beam;
  batches_ = batches;
}

void Matrix::Reshape2D()
{
  rows_ = rows_ * beam_ * batches_;
  beam_ = 1;
  batches_ = 1;
}


std::string Matrix::Debug(size_t verbosity) const
{
  std::stringstream strm;
  strm << BaseMatrix::Debug(verbosity) << " " << mem_;
  //cerr << "Debug1=" << strm.str() << endl;

  if (verbosity == 1) {
    //cerr << "Debug2" << endl;
    float sum = Sum(mem_, size(), openCLInfo_);
    //cerr << "Debug3" << endl;
    strm << " sum=" << sum << std::flush;
    //cerr << "Debug4" << endl;
  }
  //cerr << "Debug5" << endl;

  return strm.str();
}

void Matrix::Swap(Matrix &other)
{
  assert(&openCLInfo_ == &other.openCLInfo_);
  std::swap(mem_, other.mem_);
  std::swap(rows_, other.rows_);
  std::swap(cols_, other.cols_);
  std::swap(beam_, other.beam_);
  std::swap(batches_, other.batches_);
  std::swap(arrSize_, other.arrSize_);
}

void Matrix::Set(const float *data)
{
  //cerr << "Set1=" << size() << endl;
  CheckError( clEnqueueWriteBuffer(openCLInfo_.commands, mem_, CL_TRUE, 0, sizeof(float) * size(), data, 0, NULL, NULL) );
  //cerr << "Set2=" << size() << endl;
}

}
}
}
