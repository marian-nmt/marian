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
        size_t tab,
        const Search &search);

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

    virtual bool CalcBeam(BestHypsBase &bestHyps,
                          std::shared_ptr<Histories>& histories,
                          std::vector<unsigned>& beamSizes,
                          Hypotheses& prevHyps,
                          State& state,
                          State& nextState,
                          const Words &filterIndices)
    {
      assert(false);
    }

    virtual void Translate(SentencesPtr sentences)
    {
      assert(false);
    }

  protected:
    mblas::Matrix SourceContext_;
};


}
}
