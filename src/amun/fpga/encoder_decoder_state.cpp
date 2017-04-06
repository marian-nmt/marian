#include "encoder_decoder_state.h"

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

std::string EncoderDecoderState::Debug() const
{
  return "";
}

}
}

