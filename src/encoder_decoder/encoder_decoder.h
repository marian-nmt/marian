#pragma once

#include <vector>
#include <yaml-cpp/yaml.h>

#include "matrix.h"
#include "decoder/scorer.h"
#include "decoder/loader.h"
#include "dl4mt.h"
#include "decoder/god.h"

#include "common/threadpool.h"

#include "encoder_decoder/encoder_decoder_state.h"

class Encoder;
class Decoder;
class Sentence;

class EncoderDecoder : public Scorer {
  private:
    using EDState = EncoderDecoderState;

  public:
    EncoderDecoder(const std::string& name,
                   const YAML::Node& config,
                   size_t tab,
                   const Weights& model);

    virtual void Score(const State& in, Prob& prob, State& out);

    virtual State* NewState();

    virtual void BeginSentenceState(State& state);

    virtual void SetSource(const Sentence& source);

    virtual void AssembleBeamState(const State& in,
                                   const Beam& beam,
                                   State& out);

    void GetAttention(mblas::Matrix& Attention);

    size_t GetVocabSize() const;

    void Filter(const std::vector<size_t>& filterIds);

    Encoder& GetEncoder();

    Decoder& GetDecoder();

  private:
    const Weights& model_;
    std::unique_ptr<Encoder> encoder_;
    std::unique_ptr<Decoder> decoder_;

    mblas::Matrix SourceContext_;
};
