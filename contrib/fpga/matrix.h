#pragma once
#include "types-fpga.h"

class Matrix
{
public:
  Matrix(const OpenCLInfo &openCLInfo, bool rowMajor, unsigned a, unsigned b)
  {
    rowMajor_ = rowMajor;
    dim_[0] = a;
    dim_[1] = b;
    size_ = a * b;

    if (rowMajor) {
      updateStridesRowMajor();
    }
    else {
      updateStridesColMajor();
    }

    cl_int err;
    mem_ = clCreateBuffer(openCLInfo.context,  CL_MEM_READ_WRITE,  sizeof(float) * size(), NULL, &err);
    CheckError(err);
  }

  cl_mem &data()
  { return mem_; }

  const cl_mem &data() const
  { return mem_; }

  bool isRowMajor() const
  { return rowMajor_; }
 
  unsigned dim(unsigned i) const
  { return dim_[i]; }

  unsigned stride(unsigned i) const
  {  return stride_[i]; }

  unsigned size() const
  { return size_; }


  unsigned indices2Id(unsigned a, unsigned b) const
  {
    assert(a < dim(0));
    assert(b < dim(1));
  
    unsigned ind =
            a * stride(0)
          + b * stride(1);
    assert(ind < size());
    return ind;
  }

protected:
  bool rowMajor_;
  unsigned dim_[2];
  unsigned stride_[2];
  unsigned size_;
  cl_mem mem_;

  void updateStridesRowMajor()
  {
    stride_[0] = 1;
    stride_[1] = dim_[0];
  }

  void updateStridesColMajor()
  {
    stride_[0] = dim_[1];
    stride_[1] = 1;
  }

};

