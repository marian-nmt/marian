#pragma once

#include <vector>
#include <yaml-cpp/yaml.h>

#include "cpu/decoder/encoder_decoder.h"
#include "cpu/nematus/encoder.h"
#include "cpu/nematus/decoder.h"
#include "cpu/nematus/model.h"

#include "cpu/mblas/matrix.h"

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
                   size_t tab,
                   const Nematus::Weights& model);

    virtual void Decode(const State& in, State& out, const std::vector<uint>& beamSizes);

    virtual void BeginSentenceState(State& state, size_t batchSize);

    virtual void Encode(const Sentences& sources);

    virtual void AssembleBeamState(const State& in,
                                   const Beam& beam,
                                   State& out);

    void GetAttention(mblas::Matrix& Attention);
    mblas::Matrix& GetAttention();

    size_t GetVocabSize() const;

    BaseMatrix& GetProbs();

    void Filter(const std::vector<uint>& filterIds);

  protected:
    const Nematus::Weights& model_;
    std::unique_ptr<Nematus::Encoder> encoder_;
    std::unique_ptr<Nematus::Decoder> decoder_;
};


}
}
}
