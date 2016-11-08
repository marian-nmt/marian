#pragma once

#include <vector>
#include <yaml-cpp/yaml.h>

#include "common/threadpool.h"

#include "common/god.h"
#include "common/loader.h"
#include "common/scorer.h"
#include "../dl4mt/dl4mt.h"

#include "../mblas/matrix.h"

class Sentence;

namespace CPU {

class Encoder;
class Decoder;

class EncoderDecoderState : public State {
  public:
	EncoderDecoderState();

    virtual std::string Debug() const;

    CPU::mblas::Matrix& GetStates();

  	CPU::mblas::Matrix& GetEmbeddings();

    const CPU::mblas::Matrix& GetStates() const;

    const CPU::mblas::Matrix& GetEmbeddings() const;

  private:
    //EncoderDecoderState();

    CPU::mblas::Matrix states_;
    CPU::mblas::Matrix embeddings_;
};

////////////////////////////////////////////////
class EncoderDecoder : public Scorer {
  private:
    using EDState = EncoderDecoderState;

  public:
    EncoderDecoder(const std::string& name,
                   const YAML::Node& config,
                   size_t tab,
                   const Weights& model);

    virtual void Score(const State& in, BaseMatrix& prob, State& out);

    virtual State* NewState();

    virtual void BeginSentenceState(State& state);

    virtual void SetSource(const Sentence& source);

    virtual void AssembleBeamState(const State& in,
                                   const Beam& beam,
                                   State& out);

    void GetAttention(mblas::Matrix& Attention);
    mblas::Matrix& GetAttention();

    size_t GetVocabSize() const;

    BaseMatrix *CreateMatrix();

    void Filter(const std::vector<size_t>& filterIds);

    CPU::Encoder& GetEncoder();

    CPU::Decoder& GetDecoder();

  private:
    const Weights& model_;
    std::unique_ptr<CPU::Encoder> encoder_;
    std::unique_ptr<CPU::Decoder> decoder_;

    mblas::Matrix SourceContext_;
};

////////////////////////////////////////////////
class EncoderDecoderLoader : public Loader {
  public:
    EncoderDecoderLoader(const std::string name,
                         const YAML::Node& config);

    virtual void Load();

    virtual ScorerPtr NewScorer(const size_t taskId);

  private:
    std::vector<std::unique_ptr<Weights>> weights_;
};

}
