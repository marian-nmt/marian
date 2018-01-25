#include <iostream>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#ifdef __APPLE__
    #include <OpenCL/opencl.h>
#else
    #include <CL/cl.h>
#endif
#include "types-fpga.h"
#include "kernel.h"

using namespace std;

class Matrix
{
public:
  Matrix(bool rowMajor, unsigned a, unsigned b)
  {
    rowMajor_ = rowMajor;
    dim_[0] = a;
    dim_[1] = b;
    size_ = a * b;

    if (rowMajor) {

    }
    else {

    }
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

int main()
{
  cerr << "Starting..." << endl;

  OpenCLInfo openCLInfo;

  openCLInfo.context = CreateContext(100, openCLInfo.devices, openCLInfo.numDevices);
  cerr << "CreateContext done" << endl;

  openCLInfo.device = openCLInfo.devices[0];
  openCLInfo.commands = CreateCommandQueue(openCLInfo);
  cerr << "CreateCommandQueue done" << endl;

  cl_kernel kernel = CreateKernel("kernels/OutputLayer.cl", "square", openCLInfo);
  cerr << "CreateKernel done" << endl;

  Matrix W(true, 85000, 512);
  Matrix X(true, 512, 640);
  Matrix B(true, 1, 85000);
  Matrix Y(true, 85000, 640);

  cerr << "Finished" << endl;
}

