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

mblas::Matrix& EncoderDecoderState::GetStates() {
  return states_;
}

const mblas::Matrix& EncoderDecoderState::GetStates() const {
  return states_;
}

mblas::Matrix& EncoderDecoderState::GetEmbeddings() {
  return embeddings_;
}

const mblas::Matrix& EncoderDecoderState::GetEmbeddings() const {
  return embeddings_;
}

std::string EncoderDecoderState::Debug() const
{
  stringstream strm;
  strm << "states_=" << states_.Debug(1) << " embeddings_=" << embeddings_.Debug(1);
  return strm.str();
}

}
}

