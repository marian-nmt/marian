#pragma once

#include <vector>
#include <yaml-cpp/yaml.h>

#include "matrix.h"
#include "decoder/scorer.h"
#include "decoder/loader.h"
#include "dl4mt.h"
#include "decoder/god.h"

#include "common/threadpool.h"

class Encoder;
class Decoder;
class Sentence;

class EncoderDecoderState : public State {
  public:
    mblas::Matrix& GetStates();

    mblas::Matrix& GetEmbeddings();

    const mblas::Matrix& GetStates() const;

    const mblas::Matrix& GetEmbeddings() const;

  private:
    mblas::Matrix states_;
    mblas::Matrix embeddings_;
};

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

class EncoderDecoderLoader : public Loader {
  public:
    EncoderDecoderLoader(const std::string name,
                         const YAML::Node& config);

    virtual void Load();

    virtual ScorerPtr NewScorer(const size_t taskId);

  private:
    std::vector<std::unique_ptr<Weights>> weights_;
};
