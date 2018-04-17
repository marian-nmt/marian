#include "cpu/decoder/encoder_decoder_state.h"

namespace amunmt {
namespace CPU {

using EDState = EncoderDecoderState;

EncoderDecoderState::EncoderDecoderState()
{
}

std::string EncoderDecoderState::Debug(unsigned verbosity) const
{
	return CPU::mblas::Debug(states_);
}

CPU::mblas::Tensor& EncoderDecoderState::GetStates() {
  return states_;
}

CPU::mblas::Tensor& EncoderDecoderState::GetEmbeddings() {
  return embeddings_;
}

const CPU::mblas::Tensor& EncoderDecoderState::GetStates() const {
  return states_;
}

const CPU::mblas::Tensor& EncoderDecoderState::GetEmbeddings() const {
  return embeddings_;
}

}
}
