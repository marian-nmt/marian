#include <vector>
#include <fstream>
#include <iostream>
#include <cassert>
#include "kernel.h"
#include "debug-devices.h"
#include "scoped_ptrs.h"

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
  cerr << "platformIdCount=" << platformIdCount << endl;

  std::vector<cl_platform_id> platformIds (platformIdCount);
  CheckError( clGetPlatformIDs (platformIdCount, platformIds.data (), nullptr));

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

  CheckError( clGetDeviceIDs(platformIds[0], CL_DEVICE_TYPE_ALL, maxDevices, devices, &numDevices));

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

cl_kernel CreateKernel(const std::string &filePath, const std::string &kernelName, const OpenCLInfo &openCLInfo)
{
  #define MAX_SOURCE_SIZE (0x100000)
  using namespace aocl_utils;

  cl_int err;                            // error code returned from api calls

  cl_program program;                 // compute program
  cl_kernel kernel;                   // compute kernel

  // Create the compute program from the source buffer
  string str = LoadKernel(filePath);
  const char *arr[1] = {str.c_str()};

  //program = clCreateProgramWithSource(openCLInfo.context, 1, (const char **) arr, NULL, &err);

  // Load the binary.
  const char *binary_file_name = filePath.c_str();

  size_t binary_size;
  scoped_array<unsigned char> binary(loadBinaryFile(binary_file_name, &binary_size));
  if(binary == NULL) {
    CheckError(CL_INVALID_PROGRAM); //, "Failed to load binary file");
  }

  scoped_array<size_t> binary_lengths(openCLInfo.numDevices);
  scoped_array<unsigned char*> binaries(openCLInfo.numDevices);
  for (unsigned i = 0; i < openCLInfo.numDevices; ++i) {
    binary_lengths[i] = binary_size;
    binaries[i] = binary;
  }

  scoped_array<cl_int> binary_status(openCLInfo.numDevices);

  program = clCreateProgramWithBinary(
                openCLInfo.context,
                openCLInfo.numDevices,
                openCLInfo.devices,
                binary_lengths,
                (const unsigned char **) binaries.get(),
                binary_status,
                &err);
  CheckError(err);

  for(unsigned i = 0; i < openCLInfo.numDevices; ++i) {
    CheckError(binary_status[i]); //, "Failed to load binary for device");
  }

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
  kernel = clCreateKernel(program, kernelName.c_str(), &err);
  CheckError(err);
  assert(kernel);

  return kernel;
}

cl_command_queue CreateCommandQueue(const OpenCLInfo &openCLInfo)
{
  int err;                            // error code returned from api calls
  cl_command_queue commands;          // compute command queue
  // Create a command commands
  //
  commands = clCreateCommandQueue(openCLInfo.context, openCLInfo.device, 0, &err);
  CheckError(err);
  assert(commands);

  return commands;
}

// Loads a file in binary form.
unsigned char *loadBinaryFile(const char *file_name, size_t *size) {
  // Open the File
  FILE* fp;
#ifdef _WIN32
  if(fopen_s(&fp, file_name, "rb") != 0) {
    return NULL;
  }
#else
  fp = fopen(file_name, "rb");
  if(fp == 0) {
    return NULL;
  }
#endif

  // Get the size of the file
  fseek(fp, 0, SEEK_END);
  *size = ftell(fp);

  // Allocate space for the binary
  unsigned char *binary = new unsigned char[*size];

  // Go back to the file start
  rewind(fp);

  // Read the file into the binary
  if(fread((void*)binary, *size, 1, fp) == 0) {
    delete[] binary;
    fclose(fp);
    return NULL;
  }

  return binary;
}

}
}


