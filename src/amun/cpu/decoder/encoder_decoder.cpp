#include "cpu/decoder/encoder_decoder.h"

#include <vector>
#include <yaml-cpp/yaml.h>

#include "common/scorer.h"

namespace amunmt {
namespace CPU {


CPUEncoderDecoderBase::CPUEncoderDecoderBase(
    const std::string& name,
    const YAML::Node& config,
    size_t tab)
  : Scorer(name, config, tab)
{}


State* CPUEncoderDecoderBase::NewState() const {
  return new EDState();
}


}
}
