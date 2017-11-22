#pragma once

#include <vector>
#include <yaml-cpp/yaml.h>

#include "common/scorer.h"
#include "common/base_best_hyps.h"
#include "common/threadpool.h"
#include "gpu/types-gpu.h"
#include "gpu/mblas/matrix.h"
#include "gpu/mblas/handles.h"
#include "gpu/mblas/array.h"


namespace amunmt {
namespace GPU {

class EncoderDecoderState;
class Encoder;
class Decoder;
class Weights;

////////////////////////////////////////////
class EncoderDecoder : public Scorer {
  private:
    typedef EncoderDecoderState EDState;

  public:
    EncoderDecoder(const God &god,
    			   const std::string& name,
                   const YAML::Node& config,
                   size_t tab,
                   const Weights& model);

    virtual ~EncoderDecoder();

    virtual void Decode(const State& in, State& out, const std::vector<uint>& beamSizes);

    virtual State* NewState() const;

    virtual void BeginSentenceState(State& state, size_t batchSize=1);

    virtual void Encode(const Sentences& source);

    virtual void AssembleBeamState(const State& in,
                                   const Beam& beam,
                                   State& out);

    void GetAttention(mblas::Matrix& Attention);

    mblas::Matrix& GetAttention();
    virtual BaseMatrix& GetProbs();

    virtual void *GetNBest();
    virtual const BaseMatrix *GetBias() const;

    size_t GetVocabSize() const;

    void Filter(const std::vector<uint>& filterIds);

  private:
    const Weights& model_;
    std::unique_ptr<Encoder> encoder_;
    std::unique_ptr<Decoder> decoder_;
    mblas::Array<uint> indices_;
    mblas::IMatrix sentenceLengths_;
      // set in Encoder::GetContext() to length (maxSentenceLength * batchSize). 1 if it's a word, 0 otherwise

    std::unique_ptr<mblas::Matrix> SourceContext_;

    EncoderDecoder(const EncoderDecoder&) = delete;
};

}
}

