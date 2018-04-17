#pragma once

#include <vector>
#include <yaml-cpp/yaml.h>

#include "cpu/decoder/encoder_decoder.h"
#include "cpu/mblas/tensor.h"
#include "cpu/dl4mt/model.h"
#include "cpu/dl4mt/encoder.h"
#include "cpu/dl4mt/decoder.h"

namespace amunmt {

class Sentence;

namespace CPU {
namespace dl4mt {

class Encoder;
class Decoder;

class EncoderDecoder : public CPUEncoderDecoderBase {
  private:
    using EDState = EncoderDecoderState;

  public:
    EncoderDecoder(const God &god,
    			   const std::string& name,
                   const YAML::Node& config,
                   unsigned tab,
                   const Weights& model);

    virtual void Decode(
        const State& in,
        State& out,
        const std::vector<unsigned>& beamSizes);

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
    const Weights& model_;
    std::unique_ptr<Encoder> encoder_;
    std::unique_ptr<Decoder> decoder_;
};

}
}
}
