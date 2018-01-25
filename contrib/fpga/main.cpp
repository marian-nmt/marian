#include <iostream>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <vector>
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

  cl_kernel kernel = CreateKernel("kernels/OutputLayer.cl", "OutputLayer_float", openCLInfo);
  cerr << "CreateKernel done" << endl;

  Matrix<float> W(openCLInfo, true, 85000, 512);
  Matrix<float> X(openCLInfo, true, 512, 640);
  Matrix<float> B(openCLInfo, true, 1, 85000);
  Matrix<float> Y(openCLInfo, true, 85000, 640);

  vector<float> vec;
  
  vec.resize(W.size(), 3.3);
  W.Set(vec.data(), vec.size());

  vec.resize(X.size(), 21.2);
  X.Set(vec.data(), vec.size());

  vec.resize(B.size(), 9.3443);
  B.Set(vec.data(), vec.size());

  CallOpenCL("kernels/matrix_functions.cl", "sum_float", openCLInfo,
      X.data(), Y.data(), X.size());

  cerr << "Finished" << endl;
}

