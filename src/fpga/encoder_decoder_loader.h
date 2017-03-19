#include "common/loader.h"
#include "types.h"


namespace amunmt {
namespace FPGA {

class Weights;

//////////////////////////////////////////////////////////////////////
class EncoderDecoderLoader : public Loader {
public:
  EncoderDecoderLoader(const EncoderDecoderLoader&) = delete;
  EncoderDecoderLoader(const std::string name,
                       const YAML::Node& config);
  virtual ~EncoderDecoderLoader();

  virtual void Load(const God &god);

  virtual ScorerPtr NewScorer(const God &god, const DeviceInfo &deviceInfo) const;
  virtual BestHypsBasePtr GetBestHyps(const God &god) const;


protected:
  std::unique_ptr<Weights> weights_;

  cl_context context_;
  cl_uint numDevices_;
  cl_device_id devices_[100];

  void CreateContext();

  int HelloWorld(const std::string &filePath);
  int HelloWorld2();

  void DebugDevicesInfo(cl_platform_id id) const;
  void DebugDevicesInfo(cl_device_id *devices, cl_uint numDevices) const;
  void DebugDeviceInfo(cl_device_id id) const;

};

}
}

