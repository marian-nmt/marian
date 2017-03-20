#include "encoder_decoder_loader.h"
#include "encoder_decoder.h"
#include "encoder.h"
#include "decoder.h"
#include "best_hyps.h"
#include "model.h"
#include "common/god.h"
#include "kernel.h"
#include "hello_world.h"

using namespace std;

namespace amunmt {
namespace FPGA {

EncoderDecoderLoader::EncoderDecoderLoader(const std::string name,
                     const YAML::Node& config)
:Loader(name, config)
{
  cerr << "opencl start" << endl;
  context_ = CreateContext(100, devices_, numDevices_);

  cerr << "EncoderDecoderLoader1:" << endl;

  cl_command_queue commands = CreateCommandQueue(context_, devices_[0]);
  cl_kernel kernel = CreateKernel("kernels/square.cl", context_, devices_[0]);

  cerr << "EncoderDecoderLoader2:" << endl;
  HelloWorld(kernel, context_, devices_[0], commands, 1024);
  cerr << "EncoderDecoderLoader3:" << endl;
  HelloWorld(kernel, context_, devices_[0], commands, 2048);
  cerr << "EncoderDecoderLoader4:" << endl;


  clReleaseCommandQueue(commands);
  cerr << "EncoderDecoderLoader5:" << endl;
  clReleaseKernel(kernel);

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


}
}

