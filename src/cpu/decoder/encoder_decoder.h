#pragma once

#include <vector>
#include <yaml-cpp/yaml.h>

#include "common/threadpool.h"

#include "common/god.h"
#include "common/loader.h"
#include "common/scorer.h"
#include "../dl4mt/dl4mt.h"

#include "../mblas/matrix.h"

namespace amunmt {

class Sentence;

namespace CPU {

class Encoder;
class Decoder;

class EncoderDecoderState : public State {
  public:
	EncoderDecoderState();
	EncoderDecoderState(const EncoderDecoderState&) = delete;

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

    virtual void Decode(const God &god, const State& in, State& out, const std::vector<size_t>& beamSizes);

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

    CPU::Encoder& GetEncoder();

    CPU::Decoder& GetDecoder();

  private:
    const Weights& model_;
    std::unique_ptr<CPU::Encoder> encoder_;
    std::unique_ptr<CPU::Decoder> decoder_;

    mblas::Matrix SourceContext_;
};

}

}
