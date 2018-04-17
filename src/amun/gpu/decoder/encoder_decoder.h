#pragma once

#include <vector>
#include <yaml-cpp/yaml.h>

#include "common/scorer.h"
#include "common/base_best_hyps.h"
#include "common/threadpool.h"
#include "gpu/types-gpu.h"
#include "gpu/mblas/tensor.h"
#include "gpu/mblas/handles.h"
#include "gpu/mblas/vector.h"


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
                   unsigned tab,
                   const Weights& model);

    virtual ~EncoderDecoder();

    virtual void Decode(const State& in, State& out, const std::vector<unsigned>& beamSizes);

    virtual State* NewState() const;

    virtual void BeginSentenceState(State& state, unsigned batchSize=1);

    virtual void Encode(const Sentences& source);

    virtual void AssembleBeamState(const State& in,
                                   const Beam& beam,
                                   State& out);

    void GetAttention(mblas::Tensor& Attention);

    mblas::Tensor& GetAttention();
    virtual BaseTensor& GetProbs();

    virtual void *GetNBest();
    virtual const BaseTensor *GetBias() const;

    unsigned GetVocabSize() const;

    void Filter(const std::vector<unsigned>& filterIds);

  private:
    const Weights& model_;
    std::unique_ptr<Encoder> encoder_;
    std::unique_ptr<Decoder> decoder_;
    mblas::Vector<unsigned> indices_;
    std::vector<unsigned> h_sentenceLengths_;
    mblas::Vector<unsigned> sentenceLengths_;
      // set in Encoder::GetContext() to length (maxSentenceLength * batchSize). 1 if it's a word, 0 otherwise

    std::unique_ptr<mblas::Tensor> SourceContext_;

    EncoderDecoder(const EncoderDecoder&) = delete;

    void SetTensorCore();
};

}
}

