// -*- mode: c++; tab-width: 2; indent-tabs-mode: nil -*-
#include <iostream>

#include "common/god.h"
#include "common/sentences.h"

#include "encoder_decoder.h"
#include "gpu/mblas/matrix_functions.h"
#include "gpu/dl4mt/dl4mt.h"
#include "gpu/decoder/encoder_decoder_state.h"
#include "gpu/decoder/best_hyps.h"
#include "gpu/dl4mt/cellstate.h"

using namespace std;

namespace amunmt {
namespace GPU {

std::unordered_map<std::string, boost::timer::cpu_timer> timers;

EncoderDecoder::EncoderDecoder(
		const God &god,
		const std::string& name,
        const YAML::Node& config,
        size_t tab,
        const Weights& model)
  : Scorer(god, name, config, tab),
    model_(model),
    encoder_(new Encoder(model_, config)),
    decoder_(new Decoder(god, model_, config)),
    indices_(god.Get<size_t>("beam-size")),
    SourceContext_(new mblas::Matrix())
{
  BEGIN_TIMER("EncoderDecoder");
}

EncoderDecoder::~EncoderDecoder()
{
  PAUSE_TIMER("EncoderDecoder");

  if (timers.size()) {
    boost::timer::nanosecond_type encDecWall = timers["EncoderDecoder"].elapsed().wall;

    cerr << "timers:" << endl;
    for (auto iter = timers.begin(); iter != timers.end(); ++iter) {
      const boost::timer::cpu_timer &timer = iter->second;
      boost::timer::cpu_times t = timer.elapsed();
      boost::timer::nanosecond_type wallTime = t.wall;

      int percent = (float) wallTime / (float) encDecWall * 100.0f;

      cerr << iter->first << " ";

      for (int i = 0; i < ((int)35 - (int)iter->first.size()); ++i) {
        cerr << " ";
      }

      cerr << timer.format(2, "%w") << " (" << percent << ")" << endl;
    }
  }

}

void EncoderDecoder::Decode(const State& in, State& out, const std::vector<uint>& beamSizes) {
  BEGIN_TIMER("Decode");
  const EDState& edIn = in.get<EDState>();
  EDState& edOut = out.get<EDState>();

  decoder_->Decode(edOut.GetStates(),
                     edIn.GetStates(),
                     edIn.GetEmbeddings(),
                     *SourceContext_,
                     sentenceLengths_,
                     beamSizes,
                     god_.UseFusedSoftmax());
  PAUSE_TIMER("Decode");
}

State* EncoderDecoder::NewState() const {
  return new EDState();
}

void EncoderDecoder::Encode(const Sentences& source) {
  BEGIN_TIMER("Encode");
  encoder_->Encode(source, tab_, *SourceContext_, sentenceLengths_);
  //cerr << "GPU SourceContext_=" << SourceContext_.Debug(1) << endl;
  PAUSE_TIMER("Encode");
}

void EncoderDecoder::BeginSentenceState(State& state, size_t batchSize) {
  //BEGIN_TIMER("BeginSentenceState");
  EDState& edState = state.get<EDState>();
  decoder_->EmptyState(edState.GetStates(), *SourceContext_, batchSize, sentenceLengths_);

  decoder_->EmptyEmbedding(edState.GetEmbeddings(), batchSize);
  //PAUSE_TIMER("BeginSentenceState");
}


void EncoderDecoder::AssembleBeamState(const State& in,
                               const Beam& beam,
                               State& out) {
  //BEGIN_TIMER("AssembleBeamState");
  std::vector<uint> beamWords;
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

  mblas::copy(beamStateIds.data(),
      beamStateIds.size(),
      indices_.data(),
      cudaMemcpyHostToDevice);
  //cerr << "indices_=" << mblas::Debug(indices_, 2) << endl;

  CellState& outstates = edOut.GetStates();
  const CellState& instates = edIn.GetStates();

  mblas::Assemble(*(outstates.output), *(instates.output), indices_);
  if (instates.cell->size() > 0) {
    mblas::Assemble(*(outstates.cell), *(instates.cell), indices_);
  }
  //cerr << "edOut.GetStates()=" << edOut.GetStates().Debug(1) << endl;

  //cerr << "beamWords=" << Debug(beamWords, 2) << endl;
  decoder_->Lookup(edOut.GetEmbeddings(), beamWords);
  //cerr << "edOut.GetEmbeddings()=" << edOut.GetEmbeddings().Debug(1) << endl;
  //PAUSE_TIMER("AssembleBeamState");
}

void EncoderDecoder::GetAttention(mblas::Matrix& Attention) {
  decoder_->GetAttention(Attention);
}

BaseMatrix& EncoderDecoder::GetProbs() {
  return decoder_->GetProbs();
}

void *EncoderDecoder::GetNBest()
{
  return &decoder_->GetNBest();
}

const BaseMatrix *EncoderDecoder::GetBias() const
{
  return decoder_->GetBias();
}

mblas::Matrix& EncoderDecoder::GetAttention() {
  return decoder_->GetAttention();
}

size_t EncoderDecoder::GetVocabSize() const {
  return decoder_->GetVocabSize();
}

void EncoderDecoder::Filter(const std::vector<uint>& filterIds) {
  decoder_->Filter(filterIds);
}


}
}

