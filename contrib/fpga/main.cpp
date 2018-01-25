#include <iostream>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include "types-fpga.h"
#include "kernel.h"
#include "matrix.h"

using namespace std;

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

  Matrix W(openCLInfo, true, 85000, 512);
  Matrix X(openCLInfo, true, 512, 640);
  Matrix B(openCLInfo, true, 1, 85000);
  Matrix Y(openCLInfo, true, 85000, 640);

  cerr << "Finished" << endl;
}

