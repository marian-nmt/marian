#include "decoder.h"

namespace amunmt {
namespace FPGA {

void Decoder::EmptyState(mblas::Tensor& State,
                const mblas::Tensor& SourceContext,
                size_t batchSize,
                const Array<int>& batchMapping)
{
  rnn1_.InitializeState(State, SourceContext, batchSize, batchMapping);
  alignment_.Init(SourceContext);
}

void Decoder::EmptyEmbedding(mblas::Tensor& Embedding, size_t batchSize) {
  Embedding.Resize(batchSize, embeddings_.GetCols());
  mblas::Fill(Embedding, 0);
}

void Decoder::Decode(mblas::Tensor& NextState,
              const mblas::Tensor& State,
              const mblas::Tensor& Embeddings,
              const mblas::Tensor& SourceContext,
              const Array<int>& mapping,
              const std::vector<uint>& beamSizes)
{
  //std::cerr << "State=" << State.Debug(1) << std::endl;
  //std::cerr << "Embeddings=" << Embeddings.Debug(1) << std::endl;
  GetHiddenState(HiddenState_, State, Embeddings);
  //std::cerr << "HiddenState_=" << HiddenState_.Debug(1) << std::endl;

  GetAlignedSourceContext(AlignedSourceContext_, HiddenState_, SourceContext, mapping, beamSizes);
  //std::cerr << "AlignedSourceContext_=" << AlignedSourceContext_.Debug(1) << std::endl;

  GetNextState(NextState, HiddenState_, AlignedSourceContext_);
  //std::cerr << "NextState=" << NextState.Debug(1) << std::endl;

  GetProbs(NextState, Embeddings, AlignedSourceContext_);
  //std::cerr << "Probs_=" << Probs_.Debug(1) << std::endl;

}

void Decoder::GetHiddenState(mblas::Tensor& HiddenState,
                    const mblas::Tensor& PrevState,
                    const mblas::Tensor& Embedding) {
  rnn1_.GetNextState(HiddenState, PrevState, Embedding);
}

void Decoder::GetAlignedSourceContext(mblas::Tensor& AlignedSourceContext,
                              const mblas::Tensor& HiddenState,
                              const mblas::Tensor& SourceContext,
                              const Array<int>& mapping,
                              const std::vector<uint>& beamSizes)
{
  alignment_.GetAlignedSourceContext(AlignedSourceContext, HiddenState, SourceContext,
                                     mapping, beamSizes);

}

void Decoder::GetNextState(mblas::Tensor& State,
                  const mblas::Tensor& HiddenState,
                  const mblas::Tensor& AlignedSourceContext)
{
  rnn2_.GetNextState(State, HiddenState, AlignedSourceContext);
}

void Decoder::GetProbs(const mblas::Tensor& State,
              const mblas::Tensor& Embedding,
              const mblas::Tensor& AlignedSourceContext)
{
  softmax_.GetProbs(Probs_, State, Embedding, AlignedSourceContext);
}

void Decoder::Lookup(mblas::Tensor& Embedding,
            const std::vector<uint>& w)
{
  embeddings_.Lookup(Embedding, w);
}

} // namespace
}

