#include <iostream>
#include "kernel.h"
#include "types.h"

using namespace std;

namespace amunmt {
namespace FPGA {

cl_kernel CreateKernel(const std::string &filePath, const cl_context &context, const cl_device_id &device)
{
  #define MAX_SOURCE_SIZE (0x100000)

  int err;                            // error code returned from api calls

  cl_program program;                 // compute program
  cl_kernel kernel;                   // compute kernel

  // Create the compute program from the source buffer
  const char *fileName = filePath.c_str();
  //std::ifstream file(fileName);
  //std::string kernelSource((std::istreambuf_iterator<char>(file)),
  //                 std::istreambuf_iterator<char>());
  //cerr << "kernelSource=" << kernelSource << endl;

  FILE *fp = fopen(fileName, "rb");
  if (!fp) {
      fprintf(stderr, "Failed to load kernel.\n");
      exit(1);
  }
  char *source_str = (char *)malloc(MAX_SOURCE_SIZE);
  size_t source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
  fclose(fp);
  //cerr << "source_str=" << source_str << endl;

  program = clCreateProgramWithSource(context, 1, (const char **) & source_str, NULL, &err);
  if (!program)
  {
      printf("Error: Failed to create compute program!\n");
      exit(1);
  }

  // Build the program executable
  //
  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if (err != CL_SUCCESS)
  {
      size_t len;
      char buffer[2048];

      printf("Error: Failed to build program executable!\n");
      clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
      printf("%s\n", buffer);
      exit(1);
  }

  // Create the compute kernel in the program we wish to run
  //
  kernel = clCreateKernel(program, "square", &err);
  if (!kernel || err != CL_SUCCESS)
  {
      printf("Error: Failed to create compute kernel!\n");
      exit(1);
  }

  return kernel;
}

cl_command_queue CreateCommandQueue(const cl_context &context, const cl_device_id &device)
{
  int err;                            // error code returned from api calls
  cl_command_queue commands;          // compute command queue
  // Create a command commands
  //
  commands = clCreateCommandQueue(context, device, 0, &err);
  if (!commands)
  {
      printf("Error: Failed to create a command commands!\n");
      exit(1);
  }

  return commands;
}

int ExecuteKernel(cl_kernel &kernel, const cl_context &context, const cl_device_id &device, cl_command_queue &commands)
{
  #define DATA_SIZE (1024)

  int err;                            // error code returned from api calls

  float data[DATA_SIZE];              // original data set given to device
  float results[DATA_SIZE];           // results returned from device
  unsigned int correct;               // number of correct results returned

  size_t global;                      // global domain size for our calculation
  size_t local;                       // local domain size for our calculation

  cl_mem input;                       // device memory used for the input array
  cl_mem output;                      // device memory used for the output array

  // Fill our data set with random float values

  //
  int i = 0;
  unsigned int count = DATA_SIZE;
  for(i = 0; i < count; i++)
      data[i] = rand() / (float)RAND_MAX;

  // Create the input and output arrays in device memory for our calculation
  //
  input = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) * count, NULL, NULL);
  output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * count, NULL, NULL);
  if (!input || !output)
  {
      printf("Error: Failed to allocate device memory!\n");
      exit(1);
  }

  // Write our data set into the input array in device memory
  //
  err = clEnqueueWriteBuffer(commands, input, CL_TRUE, 0, sizeof(float) * count, data, 0, NULL, NULL);
  if (err != CL_SUCCESS)
  {
      printf("Error: Failed to write to source array!\n");
      exit(1);
  }

  // Set the arguments to our compute kernel
  //
  err = 0;
  err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
  err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
  err |= clSetKernelArg(kernel, 2, sizeof(unsigned int), &count);
  if (err != CL_SUCCESS)
  {
      printf("Error: Failed to set kernel arguments! %d\n", err);
      exit(1);
  }

  // Get the maximum work group size for executing the kernel on the device
  //
  err = clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
  if (err != CL_SUCCESS)
  {
      printf("Error: Failed to retrieve kernel work group info! %d\n", err);
      exit(1);
  }

  // Execute the kernel over the entire range of our 1d input data set
  // using the maximum number of work group items for this device
  //
  global = count;
  err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
  if (err)
  {
      printf("Error: Failed to execute kernel!\n");
      return EXIT_FAILURE;
  }

  // Wait for the command commands to get serviced before reading back results
  //
  clFinish(commands);

  // Read back the results from the device to verify the output
  //
  err = clEnqueueReadBuffer( commands, output, CL_TRUE, 0, sizeof(float) * count, results, 0, NULL, NULL );
  if (err != CL_SUCCESS)
  {
      printf("Error: Failed to read output array! %d\n", err);
      exit(1);
  }

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
  clReleaseMemObject(input);
  clReleaseMemObject(output);
  return 0;
}

}
}


