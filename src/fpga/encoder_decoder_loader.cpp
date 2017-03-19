#include "encoder_decoder_loader.h"
#include "encoder_decoder.h"
#include "encoder.h"
#include "decoder.h"
#include "best_hyps.h"
#include "model.h"
#include "common/god.h"
#include "kernel.h"

using namespace std;

namespace amunmt {
namespace FPGA {

EncoderDecoderLoader::EncoderDecoderLoader(const std::string name,
                     const YAML::Node& config)
:Loader(name, config)
{
  cerr << "opencl start" << endl;
  context_ = CreateContext(100, devices_);

  cerr << "HelloWorld:" << endl;

  cl_command_queue commands = CreateCommandQueue(context_, devices_[0]);
  cl_kernel kernel = CreateKernel("kernels/square.cl", context_, devices_[0]);

  HelloWorld(kernel, context_, devices_[0], commands, 1024);
  HelloWorld(kernel, context_, devices_[0], commands, 768);


  clReleaseCommandQueue(commands);
  clReleaseKernel(kernel);

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

cl_context EncoderDecoderLoader::CreateContext(
    size_t maxDevices,
    cl_device_id *devices)
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
  CL_CHECK(clGetDeviceIDs(platformIds[0], CL_DEVICE_TYPE_GPU, maxDevices, devices, &numDevices_));

  int err;
  cl_context ret = clCreateContext(NULL, 1, devices, &pfn_notify, NULL, &err);

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

