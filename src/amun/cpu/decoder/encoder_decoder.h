#pragma once

#include <yaml-cpp/yaml.h>

#include "common/scorer.h"
#include "cpu/mblas/tensor.h"
#include "cpu/decoder/encoder_decoder_state.h"

namespace amunmt {
namespace CPU {

class CPUEncoderDecoderBase : public Scorer {
  private:
    using EDState = EncoderDecoderState;

  public:
    CPUEncoderDecoderBase(
    	const God &god,
        const std::string& name,
        const YAML::Node& config,
        unsigned tab);

    virtual State* NewState() const;

    virtual void GetAttention(mblas::Tensor& Attention) = 0;
    virtual mblas::Tensor& GetAttention() = 0;

    virtual void *GetNBest()
    {
      assert(false);
      return nullptr;
    }

    virtual const BaseTensor *GetBias() const
    {
      assert(false);
      return nullptr;
    }

  protected:
    mblas::Tensor SourceContext_;
};


}
}
