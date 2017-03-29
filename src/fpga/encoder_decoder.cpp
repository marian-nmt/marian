
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
:Scorer(name, config, tab)
,model_(model)
,openCLInfo_(openCLInfo)
,sourceContext_(openCLInfo.context, openCLInfo.device)
,encoder_(new Encoder(openCLInfo, model_))
,decoder_(new Decoder(god, model_))
{

}

void EncoderDecoder::SetSource(const Sentences& sources)
{
  encoder_->GetContext(sources, tab_, sourceContext_);
  cerr << "FPGA sourceContext_=" << sourceContext_.Debug(true) << endl;
}

void EncoderDecoder::BeginSentenceState(State& state, size_t batchSize)
{
  EDState& edState = state.get<EDState>();

}

void EncoderDecoder::Decode(const God &god, const State& in,
                   State& out, const std::vector<size_t>& beamSizes)
{

}

void EncoderDecoder::AssembleBeamState(const State& in,
                               const Beam& beam,
                               State& out)
{

}

void EncoderDecoder::Filter(const std::vector<size_t>&)
{

}

State* EncoderDecoder::NewState() const
{
  return new EncoderDecoderState();
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
