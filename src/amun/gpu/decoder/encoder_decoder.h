#pragma once

#include <vector>
#include <yaml-cpp/yaml.h>

#include "common/scorer.h"
#include "common/base_best_hyps.h"
#include "common/threadpool.h"
#include "gpu/types-gpu.h"
#include "gpu/mblas/matrix.h"
#include "gpu/mblas/handles.h"
#include "gpu/mblas/vector.h"
#include "enc_out_buffer.h"


namespace amunmt {

class Histories;

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
                   const Weights& model,
                   const Search &search);

    virtual ~EncoderDecoder();

    virtual void Encode(const SentencesPtr &source);

    virtual State* NewState() const;

    void GetAttention(mblas::Matrix& Attention);

    mblas::Matrix& GetAttention();
    virtual BaseMatrix& GetProbs();

    virtual void *GetNBest();
    virtual const BaseMatrix *GetBias() const;

    unsigned GetVocabSize() const;

    void Filter(const std::vector<unsigned>& filterIds);


  private:
    const Weights& model_;
    std::unique_ptr<Encoder> encoder_;
    std::unique_ptr<Decoder> decoder_;

    EncOutBuffer encDecBuffer_;
    std::unique_ptr<std::thread> decThread_;

    EncoderDecoder(const EncoderDecoder&) = delete;


    /////////////////////////////////////////////////////////////////////////////////////
    // const-batch2
    void DecodeAsync();
    void DecodeAsyncInternal();
    void DecodeAsyncInternal(EncOutPtr encOut);

    void InitBatch(Histories &histories,
                    mblas::Vector<unsigned> &sentenceLengths,
                    mblas::Matrix &sourceContext,
                    mblas::Matrix &SCU,
                    State &state);

    void TopupBatch(Histories &histories,
                    mblas::Vector<unsigned> &sentenceLengths,
                    mblas::Matrix &sourceContext,
                    mblas::Matrix &SCU,
                    State &nextState,
                    State &state);

    void BeginSentenceState(const Histories& histories,
                            const mblas::Matrix &sourceContext,
                            const mblas::Vector<unsigned> &sentenceLengths,
                            State& state,
                            mblas::Matrix& SCU) const;

    void BeginSentenceStateTopup(const Histories& histories,
                            const mblas::Matrix &sourceContext,
                            const mblas::Vector<unsigned> &sentenceLengths,
                            State& state,
                            mblas::Matrix& SCU,
                            const std::vector<BufferOutput> &newSentences,
                            const mblas::Vector<unsigned> &d_oldBatchIds,
                            const std::vector<unsigned> &newBatchIds,
                            const mblas::Vector<unsigned> &oldHypoIds,
                            const std::vector<unsigned> &newHypoIds) const;

    void CalcBeam(BestHypsBase &bestHyps,
                    Histories& histories,
                    State& state,
                    State& nextState,
                    const Words &filterIndices);

    void AssembleBeamState(const Histories& histories,
                           const State& state,
                           State& nextState) const;

    void AssembleBeamStateTopup(const Histories& histories,
                            const State& inState,
                            const mblas::Vector<unsigned> &d_oldHypoIds,
                            State& outState) const;

};

}
}

