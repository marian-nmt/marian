#include "decoder.h"

namespace amunmt {
namespace FPGA {

void Decoder::EmptyState(mblas::Matrix& State,
                const mblas::Matrix& SourceContext,
                size_t batchSize,
                const Array<int>& batchMapping)
{
  rnn1_.InitializeState(State, SourceContext, batchSize, batchMapping);
  alignment_.Init(SourceContext);
}


}
}

