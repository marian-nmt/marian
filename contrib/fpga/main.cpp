#include <iostream>
#include <cstdio>
#include <cstdlib>
#ifdef __APPLE__
    #include <OpenCL/opencl.h>
#else
    #include <CL/cl.h>
#endif

using namespace std;

int main()
{
  cerr << "Starting..." << endl;

  cl_mem mem_;

  cerr << "Finished" << endl;
}

