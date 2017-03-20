#include <vector>
#include <fstream>
#include <iostream>
#include <cassert>
#include "kernel.h"
#include "types.h"
#include "debug-devices.h"

using namespace std;

namespace amunmt {
namespace FPGA {

cl_context CreateContext(
    size_t maxDevices,
    cl_device_id *devices,
    cl_uint &numDevices)
{
  cl_uint platformIdCount = 0;
  CheckError( clGetPlatformIDs (0, nullptr, &platformIdCount));

  std::vector<cl_platform_id> platformIds (platformIdCount);
  CheckError( clGetPlatformIDs (platformIdCount, platformIds.data (), nullptr));

  cerr << "platformIdCount=" << platformIdCount << endl;

  for (int i=0; i<platformIdCount; i++)
  {
    char buffer[10240];
    cerr << i << ":";

    CheckError( clGetPlatformInfo(platformIds[i], CL_PLATFORM_PROFILE, 10240, buffer, NULL));
    cerr << " profile=" << buffer;

    CheckError( clGetPlatformInfo(platformIds[i], CL_PLATFORM_VERSION, 10240, buffer, NULL));
    cerr << " version=" << buffer;

    CheckError( clGetPlatformInfo(platformIds[i], CL_PLATFORM_NAME, 10240, buffer, NULL));
    cerr << " name=" << buffer;

    CheckError( clGetPlatformInfo(platformIds[i], CL_PLATFORM_VENDOR, 10240, buffer, NULL));
    cerr << " vendor=" << buffer;

    CheckError( clGetPlatformInfo(platformIds[i], CL_PLATFORM_EXTENSIONS, 10240, buffer, NULL));
    cerr << " extension=" << buffer;

    DebugDevicesInfo(platformIds[i]);

    cerr << endl;
  }

  CheckError( clGetDeviceIDs(platformIds[0], CL_DEVICE_TYPE_GPU, maxDevices, devices, &numDevices));

  int err;
  cl_context ret = clCreateContext(NULL, 1, devices, &pfn_notify, NULL, &err);
  CheckError(err);

  /*
  cl_context context = clCreateContextFromType(
      0,      // platform ID
      CL_DEVICE_TYPE_GPU, // ask for a GPU
      NULL,  // error callback
      NULL,  // user data for callback
      NULL); // error code
  */
  if (!ret) {
    printf("Error: Failed to create a compute context!\n");
    abort();
  }

  return ret;
}

std::string LoadKernel(const std::string &filePath)
{
 std::ifstream in(filePath.c_str());
 std::string result (
   (std::istreambuf_iterator<char> (in)),
   std::istreambuf_iterator<char> ());
 return result;
}

cl_kernel CreateKernel(const std::string &filePath, const cl_context &context, const cl_device_id &device)
{
  #define MAX_SOURCE_SIZE (0x100000)

  int err;                            // error code returned from api calls

  cl_program program;                 // compute program
  cl_kernel kernel;                   // compute kernel

  // Create the compute program from the source buffer
  string str = LoadKernel(filePath);
  const char *arr[1] = {str.c_str()};
  program = clCreateProgramWithSource(context, 1, (const char **) arr, NULL, &err);
  CheckError(err);
  assert(program);

  // Build the program executable
  //
  CheckError( clBuildProgram(program, 0, NULL, NULL, NULL, NULL) );
  /*
  if (err != CL_SUCCESS)
  {
      size_t len;
      char buffer[2048];

      printf("Error: Failed to build program executable!\n");
      clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
      printf("%s\n", buffer);
      exit(1);
  }
  */

  // Create the compute kernel in the program we wish to run
  //
  kernel = clCreateKernel(program, "square", &err);
  CheckError(err);
  assert(kernel);

  return kernel;
}

cl_command_queue CreateCommandQueue(const cl_context &context, const cl_device_id &device)
{
  int err;                            // error code returned from api calls
  cl_command_queue commands;          // compute command queue
  // Create a command commands
  //
  commands = clCreateCommandQueue(context, device, 0, &err);
  CheckError(err);
  assert(commands);

  return commands;
}

void HelloWorld(
    cl_kernel &kernel,
    const cl_context &context,
    const cl_device_id &device,
    const cl_command_queue &commands,
    size_t dataSize)
{
  int err;                            // error code returned from api calls

  float data[dataSize];              // original data set given to device
  float results[dataSize];           // results returned from device
  unsigned int correct;               // number of correct results returned

  size_t global;                      // global domain size for our calculation
  size_t local;                       // local domain size for our calculation

  cl_mem input;                       // device memory used for the input array
  cl_mem output;                      // device memory used for the output array

  // Fill our data set with random float values
  int i = 0;
  unsigned int count = dataSize;
  for(i = 0; i < count; i++)
      data[i] = rand() / (float)RAND_MAX;

  // Create the input and output arrays in device memory for our calculation
  //
  input = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) * count, NULL, &err);
  CheckError(err);
  assert(input);

  output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * count, NULL, &err);
  CheckError(err);
  assert(output);

  // Write our data set into the input array in device memory
  //
  CheckError( clEnqueueWriteBuffer(commands, input, CL_TRUE, 0, sizeof(float) * count, data, 0, NULL, NULL) );

  // Set the arguments to our compute kernel
  CheckError( clSetKernelArg(kernel, 0, sizeof(cl_mem), &input) );
  CheckError( clSetKernelArg(kernel, 1, sizeof(cl_mem), &output) );
  CheckError( clSetKernelArg(kernel, 2, sizeof(unsigned int), &count) );

  // Get the maximum work group size for executing the kernel on the device
  //
  CheckError( clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL) );

  // Execute the kernel over the entire range of our 1d input data set
  // using the maximum number of work group items for this device
  //
  global = count;

  cerr << "dataSize=" << dataSize << endl;
  cerr << "count=" << count << endl;
  cerr << "local=" << local << endl;
  cerr << "global=" << global << endl;

  CheckError( clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL) );

  // Wait for the command commands to get serviced before reading back results
  //
  CheckError( clFinish(commands) );

  // Read back the results from the device to verify the output
  //
  CheckError( clEnqueueReadBuffer( commands, output, CL_TRUE, 0, sizeof(float) * count, results, 0, NULL, NULL ) );

  // Validate our results
  //
  correct = 0;
  for(i = 0; i < count; i++)
  {
      if(results[i] == data[i] * data[i])
          correct++;
  }

  // Print a brief summary detailing the results
  //
  cerr << "Computed " << correct << "/" << count << " correct values!\n";

  // Shutdown and cleanup
  //
  CheckError( clReleaseMemObject(input) );
  CheckError( clReleaseMemObject(output) );

}

}
}


