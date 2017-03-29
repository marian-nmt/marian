#pragma once
#include <iostream>

#ifdef __APPLE__
    #include <OpenCL/opencl.h>
#else
    #include <CL/cl.h>
#endif

inline void CheckError(cl_int error)
 {
  if (error != CL_SUCCESS) {
    std::cerr << "OpenCL call failed with error " << error << std::endl;
    std::exit (1);
  }
 }

inline void pfn_notify(const char *errinfo, const void *private_info, size_t cb, void *user_data)
{
  fprintf(stderr, "OpenCL Error (via pfn_notify): %s\n", errinfo);
}

struct OpenCLInfo
{
  cl_context context;
  cl_uint numDevices;
  cl_device_id devices[100];
  cl_device_id device;
  cl_command_queue commands;
};
