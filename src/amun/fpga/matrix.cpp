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
:dims_{0, 0, 0, 0}
,arr_(openCLInfo)
{
}

Matrix::Matrix(const OpenCLInfo &openCLInfo, size_t rows, size_t cols, bool zero)
:dims_{(uint)rows, (uint)cols, 1, 1}
,arr_(openCLInfo, rows * cols)
{
  if (zero) {
    arr_.Set(0);
  }

}

Matrix::Matrix(const OpenCLInfo &openCLInfo, size_t rows, size_t cols, float *val)
:dims_{(uint)rows, (uint)cols, 1, 1}
,arr_(openCLInfo, rows * cols, val)
{
}

Matrix::Matrix(const Matrix &other)
:dims_{other.dims_[0], other.dims_[1], other.dims_[2], other.dims_[3]}
,arr_(other.arr_)
{
}

Matrix::Matrix(Matrix &&other)
:Matrix(other.GetOpenCLInfo())
{
  Swap(other);
}


Matrix::~Matrix()
{
}

void Matrix::Resize(size_t rows, size_t cols, size_t beam, size_t batches)
{
  size_t newSize = cols * rows * beam * batches;
  arr_.resize(newSize);

  dims_[0] = rows;
  dims_[1] = cols;
  dims_[2] = beam;
  dims_[3] = batches;
}

void Matrix::Reshape(size_t rows, size_t cols, size_t beam, size_t batches)
{
  size_t newSize = cols * rows * beam * batches;
  amunmt_UTIL_THROW_IF2(newSize > arr_.size(), "Must reshape to same or smaller size");
  Resize(rows, cols, beam, batches);
}

void Matrix::Reshape2D()
{
  Reshape(dims_[0] * dims_[2] * dims_[3], dims_[1], 1, 1);
}


std::string Matrix::Debug(size_t verbosity) const
{
  std::stringstream strm;
  strm << BaseMatrix::Debug(verbosity) << " " << arr_.Debug(verbosity);
  //cerr << "Debug1=" << strm.str() << endl;

  return strm.str();
}

void Matrix::Swap(Matrix &other)
{
  assert(&arr_.GetOpenCLInfo() == &other.arr_.GetOpenCLInfo());
  std::swap(dims_, other.dims_);
  arr_.Swap(other.arr_);
}

void Matrix::Set(const float *data)
{
  arr_.Set(data, size());
}

}
}
}
