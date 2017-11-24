#include "cpu/nematus/encoder_decoder.h"

#include <vector>
#include <yaml-cpp/yaml.h>

#include "common/sentence.h"
#include "common/sentences.h"

#include "cpu/decoder/encoder_decoder_loader.h"
#include "cpu/mblas/matrix.h"

using namespace std;

namespace amunmt {
namespace CPU {
namespace Nematus {

using EDState = EncoderDecoderState;

EncoderDecoder::EncoderDecoder(const God &god,
							   const std::string& name,
                               const YAML::Node& config,
                               size_t tab,
                               const Nematus::Weights& model)
  : CPUEncoderDecoderBase(god, name, config, tab),
    model_(model),
    encoder_(new CPU::Nematus::Encoder(model_)),
    decoder_(new CPU::Nematus::Decoder(model_))
{}


void EncoderDecoder::Decode(const State& in, State& out, const std::vector<uint>&) {
  const EDState& edIn = in.get<EDState>();
  EDState& edOut = out.get<EDState>();

  decoder_->Decode(edOut.GetStates(), edIn.GetStates(),
                   edIn.GetEmbeddings(), SourceContext_);
}


void EncoderDecoder::BeginSentenceState(State& state, size_t batchSize) {
  EDState& edState = state.get<EDState>();
  decoder_->EmptyState(edState.GetStates(), SourceContext_, batchSize);
  decoder_->EmptyEmbedding(edState.GetEmbeddings(), batchSize);
}


void EncoderDecoder::Encode(const Sentences& sources) {
  encoder_->GetContext(sources.at(0)->GetWords(tab_),
                        SourceContext_);
}


void EncoderDecoder::AssembleBeamState(const State& in,
                                       const Beam& beam,
                                       State& out) {
  std::vector<uint> beamWords;
  std::vector<uint> beamStateIds;
  for(auto h : beam) {
      beamWords.push_back(h->GetWord());
      beamStateIds.push_back(h->GetPrevStateIndex());
  }

  const EDState& edIn = in.get<EDState>();
  EDState& edOut = out.get<EDState>();

  edOut.GetStates() = mblas::Assemble<mblas::byRow, mblas::Matrix>(edIn.GetStates(), beamStateIds);
  decoder_->Lookup(edOut.GetEmbeddings(), beamWords);
}


void EncoderDecoder::GetAttention(mblas::Matrix& Attention) {
  decoder_->GetAttention(Attention);
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


BaseMatrix& EncoderDecoder::GetProbs() {
  return decoder_->GetProbs();
}

}
}
}
