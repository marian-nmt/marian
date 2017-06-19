#pragma once

#include <vector>
#include <yaml-cpp/yaml.h>

#include "common/threadpool.h"

#include "common/loader.h"
#include "common/scorer.h"

#include "cpu/dl4mt/dl4mt.h"
#include "cpu/nematus/encoder.h"
#include "cpu/nematus/decoder.h"

#include "cpu/mblas/matrix.h"
#include "cpu/decoder/encoder_decoder_state.h"

namespace amunmt {

class Sentence;

namespace CPU {

namespace dl4mt {
class Encoder;
class Decoder;
}

class EncoderDecoder : public Scorer {
  private:
    using EDState = EncoderDecoderState;

  public:
    EncoderDecoder(const std::string& name,
                   const YAML::Node& config,
                   size_t tab,
                   const dl4mt::Weights& model);

    virtual void Decode(const State& in, State& out, const std::vector<size_t>& beamSizes);

    virtual State* NewState() const;

    virtual void BeginSentenceState(State& state, size_t batchSize);

    virtual void SetSource(const Sentences& sources);

    virtual void AssembleBeamState(const State& in,
                                   const Beam& beam,
                                   State& out);

    void GetAttention(mblas::Matrix& Attention);
    mblas::Matrix& GetAttention();

    size_t GetVocabSize() const;

    BaseMatrix& GetProbs();

    void Filter(const std::vector<size_t>& filterIds);

  private:
    const dl4mt::Weights& model_;
    std::unique_ptr<dl4mt::Encoder> encoder_;
    std::unique_ptr<dl4mt::Decoder> decoder_;

    mblas::Matrix SourceContext_;
};

class NematusEncoderDecoder : public Scorer {
  private:
    using EDState = EncoderDecoderState;

  public:
    NematusEncoderDecoder(const std::string& name,
                   const YAML::Node& config,
                   size_t tab,
                   const Nematus::Weights& model);

    virtual void Decode(const State& in, State& out, const std::vector<size_t>& beamSizes);

    virtual State* NewState() const;

    virtual void BeginSentenceState(State& state, size_t batchSize);

    virtual void SetSource(const Sentences& sources);

    virtual void AssembleBeamState(const State& in,
                                   const Beam& beam,
                                   State& out);

    void GetAttention(mblas::Matrix& Attention);
    mblas::Matrix& GetAttention();

    size_t GetVocabSize() const;

    BaseMatrix& GetProbs();

    void Filter(const std::vector<size_t>& filterIds);

  private:
    const Nematus::Weights& model_;
    std::unique_ptr<Nematus::Encoder> encoder_;
    std::unique_ptr<Nematus::Decoder> decoder_;

    mblas::Matrix SourceContext_;
};

}
}
