#pragma once

#include <vector>
#include <yaml-cpp/yaml.h>

#include "common/scorer.h"
#include "common/loader.h"
#include "gpu/dl4mt/dl4mt.h"

#include "common/threadpool.h"

namespace GPU {

class EncoderDecoderState : public State {
  public:
    virtual std::string Debug() const;

    mblas::Matrix& GetStates();
    mblas::Matrix& GetEmbeddings();
    const mblas::Matrix& GetStates() const;
    const mblas::Matrix& GetEmbeddings() const;

  private:
    mblas::Matrix states_;
    mblas::Matrix embeddings_;
};

////////////////////////////////////////////
class EncoderDecoder : public Scorer {
  private:
    typedef EncoderDecoderState EDState;

  public:
    EncoderDecoder(const std::string& name,
                   const YAML::Node& config,
                   size_t tab,
                   const Weights& model);

    virtual void Score(const State& in,
    		BaseMatrix& prob,
    		State& out);

    virtual State* NewState();

    virtual void BeginSentenceState(State& state);

    virtual void SetSource(const Sentence& source);

    virtual void AssembleBeamState(const State& in,
                                   const Beam& beam,
                                   State& out);

    void GetAttention(mblas::Matrix& Attention);

    size_t GetVocabSize() const;

    void Filter(const std::vector<size_t>& filterIds) {
      decoder_->Filter(filterIds);
    }

    virtual BaseMatrix *CreateMatrix();

  private:
    const Weights& model_;
    std::unique_ptr<Encoder> encoder_;
    std::unique_ptr<Decoder> decoder_;

    mblas::Matrix SourceContext_;
};

////////////////////////////////////////////
class EncoderDecoderLoader : public Loader {
  public:
    EncoderDecoderLoader(const std::string name,
                         const YAML::Node& config);

    virtual void Load();

    virtual ScorerPtr NewScorer(size_t taskId);

  private:
    std::vector<std::unique_ptr<Weights>> weights_;
};

}
