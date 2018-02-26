#include <sstream>
#include "encoder_decoder_state.h"

using namespace std;

namespace amunmt {
namespace FPGA {

EncoderDecoderState::EncoderDecoderState(const OpenCLInfo &openCLInfo)
:states_(openCLInfo)
,embeddings_(openCLInfo)
{

}

mblas::Tensor& EncoderDecoderState::GetStates() {
  return states_;
}

const mblas::Tensor& EncoderDecoderState::GetStates() const {
  return states_;
}

mblas::Tensor& EncoderDecoderState::GetEmbeddings() {
  return embeddings_;
}

const mblas::Tensor& EncoderDecoderState::GetEmbeddings() const {
  return embeddings_;
}

std::string EncoderDecoderState::Debug(unsigned verbosity) const
{
  stringstream strm;
  strm << "states_=" << states_.Debug(verbosity) << " embeddings_=" << embeddings_.Debug(verbosity);
  return strm.str();
}

}
}

