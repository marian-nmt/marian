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

int main()
{
  cerr << "Starting..." << endl;

  OpenCLInfo openCLInfo_;

  openCLInfo_.context = CreateContext(100, openCLInfo_.devices, openCLInfo_.numDevices);
  cerr << "CreateContext done" << endl;

  openCLInfo_.device = openCLInfo_.devices[0];
  openCLInfo_.commands = CreateCommandQueue(openCLInfo_);
  cerr << "CreateCommandQueue done" << endl;

  cl_kernel kernel = CreateKernel("kernels/OutputLayer.cl", "square", openCLInfo_);
  cerr << "CreateKernel done" << endl;

  cl_mem mem_;

  cerr << "Finished" << endl;
}

