#include "common/loader.h"
#include "types-fpga.h"


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
  virtual BestHypsBasePtr GetBestHyps(const God &god, const DeviceInfo &deviceInfo) const;


protected:
  std::unique_ptr<Weights> weights_;

  OpenCLInfo openCLInfo_;

  //int HelloWorld(const std::string &filePath);
  int HelloWorld2();

};

}
}

