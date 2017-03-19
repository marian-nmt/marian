#include "encoder_decoder_loader.h"
#include "encoder_decoder.h"
#include "encoder.h"
#include "decoder.h"
#include "best_hyps.h"
#include "model.h"
#include "common/god.h"

using namespace std;

namespace amunmt {
namespace FPGA {

const char *KernelSource = "\n" \
"__kernel void square(                                                       \n" \
"   __global float* input,                                              \n" \
"   __global float* output,                                             \n" \
"   const unsigned int count)                                           \n" \
"{                                                                      \n" \
"   unsigned int i = get_global_id(0);                                           \n" \
"   if(i < count)                                                       \n" \
"       output[i] = input[i] * input[i];                                \n" \
"}                                                                      \n" \
"\n";

EncoderDecoderLoader::EncoderDecoderLoader(const std::string name,
                     const YAML::Node& config)
:Loader(name, config)
{
  cerr << "opencl start" << endl;
  CreateContext();

  cerr << "HelloWorld:" << endl;
  HelloWorld();

  cerr << "HelloWorld2:" << endl;
  HelloWorld2();

  cerr << "opencl end" << endl;
}



EncoderDecoderLoader::~EncoderDecoderLoader()
{
  clReleaseContext(context_);
}

void EncoderDecoderLoader::Load(const God &god)
{
  std::string path = Get<std::string>("path");
  std::vector<size_t> devices = god.Get<std::vector<size_t>>("devices");

  size_t d = 0;

  Weights *weights = new Weights(context_, path, d);
  weights_.reset(weights);
}

ScorerPtr EncoderDecoderLoader::NewScorer(const God &god, const DeviceInfo &deviceInfo) const
{

  size_t d = deviceInfo.deviceId;
  size_t tab = Has("tab") ? Get<size_t>("tab") : 0;

  EncoderDecoder *ed = new EncoderDecoder(god, name_, config_, tab, *weights_, context_);
  return ScorerPtr(ed);
}

BestHypsBasePtr EncoderDecoderLoader::GetBestHyps(const God &god) const
{
  BestHyps *obj = new BestHyps();
  return BestHypsBasePtr(obj);
}

void EncoderDecoderLoader::CreateContext()
{
  cl_uint platformIdCount = 0;
  CL_CHECK(clGetPlatformIDs (0, nullptr, &platformIdCount));

  std::vector<cl_platform_id> platformIds (platformIdCount);
  CL_CHECK(clGetPlatformIDs (platformIdCount, platformIds.data (), nullptr));

  cerr << "platformIdCount=" << platformIdCount << endl;

  for (int i=0; i<platformIdCount; i++)
  {
    char buffer[10240];
    cerr << i << ":";

    CL_CHECK(clGetPlatformInfo(platformIds[i], CL_PLATFORM_PROFILE, 10240, buffer, NULL));
    cerr << " profile=" << buffer;

    CL_CHECK(clGetPlatformInfo(platformIds[i], CL_PLATFORM_VERSION, 10240, buffer, NULL));
    cerr << " version=" << buffer;

    CL_CHECK(clGetPlatformInfo(platformIds[i], CL_PLATFORM_NAME, 10240, buffer, NULL));
    cerr << " name=" << buffer;

    CL_CHECK(clGetPlatformInfo(platformIds[i], CL_PLATFORM_VENDOR, 10240, buffer, NULL));
    cerr << " vendor=" << buffer;

    CL_CHECK(clGetPlatformInfo(platformIds[i], CL_PLATFORM_EXTENSIONS, 10240, buffer, NULL));
    cerr << " extension=" << buffer;

    DebugDevicesInfo(platformIds[i]);

    cerr << endl;
  }

  // CL_CHECK(clGetDeviceIDs(NULL, CL_DEVICE_TYPE_ALL, 100, devices, &numDevices));
  CL_CHECK(clGetDeviceIDs(platformIds[0], CL_DEVICE_TYPE_GPU, 100, devices_, &numDevices_));

  int err;
  context_ = clCreateContext(NULL, 1, devices_, &pfn_notify, NULL, &err);

  /*
  cl_context context = clCreateContextFromType(
      0,      // platform ID
      CL_DEVICE_TYPE_GPU, // ask for a GPU
      NULL,  // error callback
      NULL,  // user data for callback
      NULL); // error code
  */
  if (!context_) {
    printf("Error: Failed to create a compute context!\n");
    abort();
  }

}

std::string LoadKernel (const char* name)
{
 std::ifstream in (name);
 std::string result (
   (std::istreambuf_iterator<char> (in)),
   std::istreambuf_iterator<char> ());
 return result;
}

cl_program CreateProgram(const std::string& source,
 cl_context context)
{
 size_t lengths [1] = { source.size () };
 const char* sources [1] = { source.data () };

 cl_int error = 0;
 cl_program program = clCreateProgramWithSource (context, 1, sources, lengths, &error);
 CheckError (error);

 return program;
}

int EncoderDecoderLoader::HelloWorld()
{
  #define DATA_SIZE (1024)
  #define MAX_SOURCE_SIZE (0x100000)

  int err;                            // error code returned from api calls

  float data[DATA_SIZE];              // original data set given to device
  float results[DATA_SIZE];           // results returned from device
  unsigned int correct;               // number of correct results returned

  size_t global;                      // global domain size for our calculation
  size_t local;                       // local domain size for our calculation

  cl_command_queue commands;          // compute command queue
  cl_program program;                 // compute program
  cl_kernel kernel;                   // compute kernel

  cl_mem input;                       // device memory used for the input array
  cl_mem output;                      // device memory used for the output array

  // Fill our data set with random float values

  //
  int i = 0;
  unsigned int count = DATA_SIZE;
  for(i = 0; i < count; i++)
      data[i] = rand() / (float)RAND_MAX;

  // Create a command commands
  //
  commands = clCreateCommandQueue(context_, devices_[0], 0, &err);
  if (!commands)
  {
      printf("Error: Failed to create a command commands!\n");
      return EXIT_FAILURE;
  }

  // Create the compute program from the source buffer
  const char fileName[] = "kernels/square.cl";
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

  program = clCreateProgramWithSource(context_, 1, (const char **) & source_str, NULL, &err);
  if (!program)
  {
      printf("Error: Failed to create compute program!\n");
      return EXIT_FAILURE;
  }

  // Build the program executable
  //
  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if (err != CL_SUCCESS)
  {
      size_t len;
      char buffer[2048];

      printf("Error: Failed to build program executable!\n");
      clGetProgramBuildInfo(program, devices_[0], CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
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

  // Create the input and output arrays in device memory for our calculation
  //
  input = clCreateBuffer(context_,  CL_MEM_READ_ONLY,  sizeof(float) * count, NULL, NULL);
  output = clCreateBuffer(context_, CL_MEM_WRITE_ONLY, sizeof(float) * count, NULL, NULL);
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
  err = clGetKernelWorkGroupInfo(kernel, devices_[0], CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
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
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(commands);
  return 0;
}

int EncoderDecoderLoader::HelloWorld2()
{
  #define DATA_SIZE (1024)

  int err;                            // error code returned from api calls

  float data[DATA_SIZE];              // original data set given to device
  float results[DATA_SIZE];           // results returned from device
  unsigned int correct;               // number of correct results returned

  size_t global;                      // global domain size for our calculation
  size_t local;                       // local domain size for our calculation

  cl_command_queue commands;          // compute command queue
  cl_program program;                 // compute program
  cl_kernel kernel;                   // compute kernel

  cl_mem input;                       // device memory used for the input array
  cl_mem output;                      // device memory used for the output array

  // Fill our data set with random float values

  //
  int i = 0;
  unsigned int count = DATA_SIZE;
  for(i = 0; i < count; i++)
      data[i] = rand() / (float)RAND_MAX;

  // Create a command commands
  //
  commands = clCreateCommandQueue(context_, devices_[0], 0, &err);
  if (!commands)
  {
      printf("Error: Failed to create a command commands!\n");
      return EXIT_FAILURE;
  }

  // Create the compute program from the source buffer
  //
  program = clCreateProgramWithSource(context_, 1, (const char **) & KernelSource, NULL, &err);
  if (!program)
  {
      printf("Error: Failed to create compute program!\n");
      return EXIT_FAILURE;
  }

  // Build the program executable
  //
  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if (err != CL_SUCCESS)
  {
      size_t len;
      char buffer[2048];

      printf("Error: Failed to build program executable!\n");
      clGetProgramBuildInfo(program, devices_[0], CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
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

  // Create the input and output arrays in device memory for our calculation
  //
  input = clCreateBuffer(context_,  CL_MEM_READ_ONLY,  sizeof(float) * count, NULL, NULL);
  output = clCreateBuffer(context_, CL_MEM_WRITE_ONLY, sizeof(float) * count, NULL, NULL);
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
  err = clGetKernelWorkGroupInfo(kernel, devices_[0], CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
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
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(commands);
  return 0;
}


void EncoderDecoderLoader::DebugDevicesInfo(cl_platform_id id) const
{
  cl_device_id devices[100];
  cl_uint numDevices = 0;
  // CL_CHECK(clGetDeviceIDs(NULL, CL_DEVICE_TYPE_ALL, 100, devices, &numDevices));
  CL_CHECK(clGetDeviceIDs(id, CL_DEVICE_TYPE_GPU, 100, devices, &numDevices));
  DebugDevicesInfo(devices, numDevices);
}

void EncoderDecoderLoader::DebugDevicesInfo(cl_device_id *devices, cl_uint numDevices) const
{
  cerr << "numDevices=" << numDevices << endl;

  for (int i=0; i<numDevices; i++)
  {
    DebugDeviceInfo(devices[i]);
  }

}

void EncoderDecoderLoader::DebugDeviceInfo(cl_device_id id) const
{
  char buffer[10240];
  cl_uint buf_uint;
  cl_ulong buf_ulong;
  cerr << id << ":";

  CL_CHECK(clGetDeviceInfo(id, CL_DEVICE_NAME, sizeof(buffer), buffer, NULL));
  cerr << " extension=" << buffer;

  CL_CHECK(clGetDeviceInfo(id, CL_DEVICE_VENDOR, sizeof(buffer), buffer, NULL));
  cerr << " vendor=" << buffer;

  CL_CHECK(clGetDeviceInfo(id, CL_DEVICE_VERSION, sizeof(buffer), buffer, NULL));
  cerr << " version=" << buffer;

  CL_CHECK(clGetDeviceInfo(id, CL_DRIVER_VERSION, sizeof(buffer), buffer, NULL));
  cerr << " driver version=" << buffer;

  CL_CHECK(clGetDeviceInfo(id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(buf_uint), &buf_uint, NULL));
  cerr << " compute units=" << buf_uint;

  CL_CHECK(clGetDeviceInfo(id, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(buf_uint), &buf_uint, NULL));
  cerr << " clock freq=" << buf_uint;

  CL_CHECK(clGetDeviceInfo(id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(buf_ulong), &buf_ulong, NULL));
  cerr << " global mem=" << buf_ulong;

  cerr << endl;

}

}
}

