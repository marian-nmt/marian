#pragma once

#include <yaml-cpp/yaml.h>

#include "common/scorer.h"
#include "cpu/mblas/matrix.h"
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
        size_t tab);

    virtual State* NewState() const;

    virtual void GetAttention(mblas::Matrix& Attention) = 0;
    virtual mblas::Matrix& GetAttention() = 0;

    virtual void *GetNBest()
    {
      assert(false);
      return nullptr;
    }

    virtual const BaseMatrix *GetBias() const
    {
      assert(false);
      return nullptr;
    }

  protected:
    mblas::Matrix SourceContext_;
};


}
}
