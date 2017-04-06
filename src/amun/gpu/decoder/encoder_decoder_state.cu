#include "gpu/decoder/encoder_decoder_state.h"

namespace amunmt {
namespace GPU {

////////////////////////////////////////////
std::string EncoderDecoderState::Debug() const
{
  return states_.Debug();
}

mblas::Matrix& EncoderDecoderState::GetStates() {
  return states_;
}

mblas::Matrix& EncoderDecoderState::GetEmbeddings() {
  return embeddings_;
}

const mblas::Matrix& EncoderDecoderState::GetStates() const {
  return states_;
}

const mblas::Matrix& EncoderDecoderState::GetEmbeddings() const {
  return embeddings_;
}

}
}

