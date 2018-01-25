#pragma once
#include <string>
#include "types-fpga.h"

cl_context CreateContext(
    size_t maxDevices,
    cl_device_id *devices,
    cl_uint &numDevices);

cl_kernel CreateKernel(const std::string &filePath, const std::string &kernelName, const OpenCLInfo &openCLInfo);
cl_command_queue CreateCommandQueue(const OpenCLInfo &openCLInfo);

/////////////////////////////////////////////////////////////////////////////////////
template <typename T>
void SetKernelArg(cl_kernel kernel, cl_uint argNum, const T &t)
{
  //std::cerr << "arg" << argNum << "=" << t << std::endl ;
  CheckError( clSetKernelArg(kernel, argNum, sizeof(T), &t) );
}

template<typename T, typename... Args>
void SetKernelArg(cl_kernel kernel, cl_uint argNum, const T &t, Args... args) // recursive variadic function
{
  //std::cerr << "arg" << argNum << "=" << t << std::endl ;
  CheckError( clSetKernelArg(kernel, argNum, sizeof(T), &t) );

  SetKernelArg(kernel, argNum + 1, args...) ;
}

template<typename... Args>
void CallOpenCL(
    const std::string &filePath,
    const std::string &kernelName,
    const OpenCLInfo &openCLInfo,
    Args... args
    )
{
  cl_int err;
  size_t global;                      // global domain size for our calculation
  size_t local;                       // local domain size for our calculation

  cl_mem output = clCreateBuffer(openCLInfo.context, CL_MEM_WRITE_ONLY, sizeof(size_t), NULL, &err);
  CheckError(err);
  assert(output);

  // create kernel
  cl_kernel kernel = CreateKernel(filePath, kernelName, openCLInfo);

  // Set the arguments to our compute kernel
  SetKernelArg(kernel, 0, args...);

  // Get the maximum work group size for executing the kernel on the device
  //
  CheckError( clGetKernelWorkGroupInfo(kernel, openCLInfo.device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL) );

  //global = 1024;
  local = 1;
  global = 1;

  //cerr << "local=" << local << endl;
  //cerr << "global=" << global << endl;

  CheckError( clEnqueueNDRangeKernel(openCLInfo.commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL) );

  // Wait for the command commands to get serviced before reading back results
  //
  CheckError( clFinish(openCLInfo.commands) );

}

/////////////////////////////////////////////////////////////////////////////////////


