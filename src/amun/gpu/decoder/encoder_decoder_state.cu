#include <sstream>
#include "gpu/decoder/encoder_decoder_state.h"

using namespace std;

namespace amunmt {
namespace GPU {

////////////////////////////////////////////
std::string EncoderDecoderState::Debug(size_t verbosity) const
{
  stringstream strm;
  strm << "states_=" << states_.Debug(verbosity) << " embeddings_=" << embeddings_.Debug(verbosity);
  return strm.str();
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

