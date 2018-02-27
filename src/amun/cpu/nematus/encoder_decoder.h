#pragma once

#include <vector>
#include <yaml-cpp/yaml.h>

#include "cpu/decoder/encoder_decoder.h"
#include "cpu/nematus/encoder.h"
#include "cpu/nematus/decoder.h"
#include "cpu/nematus/model.h"

#include "cpu/mblas/tensor.h"

namespace amunmt {

class Sentence;

namespace CPU {
namespace Nematus {

class EncoderDecoder : public CPUEncoderDecoderBase {
  private:
    using EDState = EncoderDecoderState;

  public:
    EncoderDecoder(const God &god,
    			   const std::string& name,
                   const YAML::Node& config,
                   unsigned tab,
                   const Nematus::Weights& model);

    virtual void Decode(const State& in, State& out, const std::vector<unsigned>& beamSizes);

    virtual void BeginSentenceState(State& state, unsigned batchSize);

    virtual void Encode(const Sentences& sources);

    virtual void AssembleBeamState(const State& in,
                                   const Beam& beam,
                                   State& out);

    void GetAttention(mblas::Tensor& Attention);
    mblas::Tensor& GetAttention();

    unsigned GetVocabSize() const;

    BaseTensor& GetProbs();

    void Filter(const std::vector<unsigned>& filterIds);

  protected:
    const Nematus::Weights& model_;
    std::unique_ptr<Nematus::Encoder> encoder_;
    std::unique_ptr<Nematus::Decoder> decoder_;
};


}
}
}
