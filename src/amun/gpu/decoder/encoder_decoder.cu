// -*- mode: c++; tab-width: 2; indent-tabs-mode: nil -*-
#include <iostream>

#include "common/god.h"
#include "common/sentences.h"

#include "encoder_decoder.h"
#include "gpu/mblas/matrix_functions.h"
#include "gpu/dl4mt/dl4mt.h"
#include "gpu/decoder/encoder_decoder_state.h"
#include "gpu/decoder/best_hyps.h"

using namespace std;

namespace amunmt {
namespace GPU {

EncoderDecoder::EncoderDecoder(
		const God &god,
		const std::string& name,
        const YAML::Node& config,
        size_t tab,
        const Weights& model)
  : Scorer(god, name, config, tab),
    model_(model),
    encoder_(new Encoder(model_)),
    decoder_(new Decoder(god, model_)),
    indices_(god.Get<size_t>("beam-size")),
    SourceContext_(new mblas::Matrix())
{
  BEGIN_TIMER("EncoderDecoder");
}

EncoderDecoder::~EncoderDecoder()
{
  PAUSE_TIMER("EncoderDecoder");
}

void EncoderDecoder::Decode(const State& in, State& out, const std::vector<uint>& beamSizes) {
  BEGIN_TIMER("Decode");
  const EDState& edIn = in.get<EDState>();
  EDState& edOut = out.get<EDState>();

  decoder_->Decode(edOut.GetStates(),
                     edIn.GetStates(),
                     edIn.GetEmbeddings(),
                     *SourceContext_,
                     sentencesMask_,
                     beamSizes);
  PAUSE_TIMER("Decode");
}

State* EncoderDecoder::NewState() const {
  return new EDState();
}

void EncoderDecoder::Encode(const Sentences& source) {
  BEGIN_TIMER("Encode");
  encoder_->Encode(source, tab_, *SourceContext_, sentencesMask_);
  //cerr << "GPU SourceContext_=" << SourceContext_.Debug(1) << endl;
  PAUSE_TIMER("Encode");
}

void EncoderDecoder::BeginSentenceState(State& state, size_t batchSize) {
  BEGIN_TIMER("BeginSentenceState");
  EDState& edState = state.get<EDState>();
  decoder_->EmptyState(edState.GetStates(), *SourceContext_, batchSize, sentencesMask_);

  decoder_->EmptyEmbedding(edState.GetEmbeddings(), batchSize);
  PAUSE_TIMER("BeginSentenceState");
}


void EncoderDecoder::AssembleBeamState(const State& in,
                               const Beam& beam,
                               State& out) {
  BEGIN_TIMER("AssembleBeamState");
  std::vector<size_t> beamWords;
  std::vector<uint> beamStateIds;
  for (const HypothesisPtr &h : beam) {
     beamWords.push_back(h->GetWord());
     beamStateIds.push_back(h->GetPrevStateIndex());
  }
  //cerr << "beamWords=" << Debug(beamWords, 2) << endl;
  //cerr << "beamStateIds=" << Debug(beamStateIds, 2) << endl;

  const EDState& edIn = in.get<EDState>();
  EDState& edOut = out.get<EDState>();
  indices_.resize(beamStateIds.size());
  HostVector<uint> tmp = beamStateIds;

  mblas::copy(thrust::raw_pointer_cast(tmp.data()),
      beamStateIds.size(),
      thrust::raw_pointer_cast(indices_.data()),
      cudaMemcpyHostToDevice);
  //cerr << "indices_=" << mblas::Debug(indices_, 2) << endl;

  mblas::Assemble(edOut.GetStates(), edIn.GetStates(), indices_);
  //cerr << "edOut.GetStates()=" << edOut.GetStates().Debug(1) << endl;

  //cerr << "beamWords=" << Debug(beamWords, 2) << endl;
  decoder_->Lookup(edOut.GetEmbeddings(), beamWords);
  //cerr << "edOut.GetEmbeddings()=" << edOut.GetEmbeddings().Debug(1) << endl;
  PAUSE_TIMER("AssembleBeamState");
}

void EncoderDecoder::GetAttention(mblas::Matrix& Attention) {
  decoder_->GetAttention(Attention);
}

BaseMatrix& EncoderDecoder::GetProbs() {
  return decoder_->GetProbs();
}

mblas::Matrix& EncoderDecoder::GetAttention() {
  return decoder_->GetAttention();
}

size_t EncoderDecoder::GetVocabSize() const {
  return decoder_->GetVocabSize();
}

void EncoderDecoder::Filter(const std::vector<size_t>& filterIds) {
  decoder_->Filter(filterIds);
}


}
}

