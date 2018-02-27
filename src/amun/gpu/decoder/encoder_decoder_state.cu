#include <sstream>
#include "gpu/decoder/encoder_decoder_state.h"
#include "gpu/dl4mt/cellstate.h"

using namespace std;

namespace amunmt {
namespace GPU {

////////////////////////////////////////////
std::string EncoderDecoderState::Debug(unsigned verbosity) const
{
  stringstream strm;
  strm << "states_.output=" << states_.output->Debug(verbosity);
  strm << " states_.cell=" << (states_.cell == NULL ? "NULL" : states_.cell->Debug(verbosity));
  strm << " embeddings_=" << embeddings_.Debug(verbosity);
  return strm.str();
}

CellState& EncoderDecoderState::GetStates() {
  return states_;
}

mblas::Tensor& EncoderDecoderState::GetEmbeddings() {
  return embeddings_;
}

const CellState& EncoderDecoderState::GetStates() const {
  return states_;
}

const mblas::Tensor& EncoderDecoderState::GetEmbeddings() const {
  return embeddings_;
}

}
}

