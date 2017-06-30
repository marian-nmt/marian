#include "cpu/decoder/encoder_decoder_state.h"

namespace amunmt {
namespace CPU {

using EDState = EncoderDecoderState;

EncoderDecoderState::EncoderDecoderState()
{
}

std::string EncoderDecoderState::Debug(size_t verbosity) const
{
	return CPU::mblas::Debug(states_);
}

CPU::mblas::Matrix& EncoderDecoderState::GetStates() {
  return states_;
}

CPU::mblas::Matrix& EncoderDecoderState::GetEmbeddings() {
  return embeddings_;
}

const CPU::mblas::Matrix& EncoderDecoderState::GetStates() const {
  return states_;
}

const CPU::mblas::Matrix& EncoderDecoderState::GetEmbeddings() const {
  return embeddings_;
}

}
}
