
#include "encoder_decoder.h"
#include "encoder_decoder_state.h"
#include "best_hyps.h"
#include "encoder.h"
#include "decoder.h"
#include "model.h"
#include "common/god.h"

using namespace std;

// Simple compute kernel which computes the square of an input array

//

namespace amunmt {
namespace FPGA {

EncoderDecoder::EncoderDecoder(
        const God &god,
        const std::string& name,
        const YAML::Node& config,
        size_t tab,
        const Weights& model,
        const OpenCLInfo &openCLInfo)
:Scorer(god, name, config, tab)
,model_(model)
,openCLInfo_(openCLInfo)
,sourceContext_(openCLInfo)
,encoder_(new Encoder(openCLInfo, model_))
,decoder_(new Decoder(openCLInfo, god, model_))
,indices_(openCLInfo)
,batchMapping_(openCLInfo)
{

}

void EncoderDecoder::Encode(const Sentences& sources)
{
  encoder_->Encode(sources, tab_, sourceContext_, batchMapping_);
  //cerr << "FPGA sourceContext_=" << sourceContext_.Debug(1) << endl;
}

void EncoderDecoder::BeginSentenceState(State& state, size_t batchSize)
{
  EDState& edState = state.get<EDState>();
  decoder_->EmptyState(edState.GetStates(), sourceContext_, batchSize, batchMapping_);

  decoder_->EmptyEmbedding(edState.GetEmbeddings(), batchSize);
}

void EncoderDecoder::Decode(const State& in,
                   State& out, const std::vector<uint>& beamSizes)
{
  const EDState& edIn = in.get<EDState>();
  EDState& edOut = out.get<EDState>();

  decoder_->Decode(edOut.GetStates(),
                     edIn.GetStates(),
                     edIn.GetEmbeddings(),
                     sourceContext_,
                     batchMapping_,
                     beamSizes);
}

void EncoderDecoder::AssembleBeamState(const State& in,
                               const Beam& beam,
                               State& out)
{
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

  indices_.Set(beamStateIds);
  //cerr << "indices_=" << indices_.Debug(2) << endl;

  mblas::Assemble(edOut.GetStates(), edIn.GetStates(), indices_);
  //cerr << "edOut.GetStates()=" << edOut.GetStates().Debug(1) << endl;

  //cerr << "beamWords=" << Debug(beamWords, 2) << endl;
  decoder_->Lookup(edOut.GetEmbeddings(), beamWords);
  //cerr << "edOut.GetEmbeddings()=" << edOut.GetEmbeddings().Debug(1) << endl;
}

void EncoderDecoder::Filter(const std::vector<uint>&)
{

}

State* EncoderDecoder::NewState() const
{
  return new EncoderDecoderState(openCLInfo_);
}

size_t EncoderDecoder::GetVocabSize() const
{
  return decoder_->GetVocabSize();
}

BaseMatrix& EncoderDecoder::GetProbs()
{
  return decoder_->GetProbs();
}

//////////////////////////////////////////////////////////////////////


}
}
