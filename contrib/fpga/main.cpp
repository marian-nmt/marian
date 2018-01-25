#include <iostream>
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
  Matrix(unsigned a, unsigned b, bool rowMajor)
  {
    dim_[0] = a;
    dim_[1] = b;
    rowMajor_ = rowMajor;
  }

  cl_mem &data()
  { return mem_; }

  const cl_mem &data() const
  { return mem_; }

  bool isRowMajor() const
  { return rowMajor_; }
 
protected:
  unsigned dim_[2];
  cl_mem mem_;
  bool rowMajor_;

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

  Matrix W(85000, 512, true);
  Matrix X(512, 640, true);
  Matrix B(1, 85000, true);
  Matrix Y(85000, 640, true);

  cerr << "Finished" << endl;
}

