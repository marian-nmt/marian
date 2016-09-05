#include "decoder.h"

//////////////////////////////////////////////////////////////
template<class Weights>
Decoder::Embeddings<Weights>::Embeddings(const Weights& model)
: w_(model)
{}

template<class Weights>
void Decoder::Embeddings<Weights>::Lookup(mblas::Matrix& Rows, const std::vector<size_t>& ids) {
  using namespace mblas;
  std::vector<size_t> tids = ids;
  for(auto&& id : tids)
	if(id >= w_.E_.rows())
	  id = 1;
  Rows = Assemble<byRow, Matrix>(w_.E_, tids);
}

template<class Weights>
size_t Decoder::Embeddings<Weights>::GetCols() {
  return w_.E_.columns();
}

template<class Weights>
size_t Decoder::Embeddings<Weights>::GetRows() const {
  return w_.E_.rows();
}

//////////////////////////////////////////////////////////////
template <class Weights1, class Weights2>
Decoder::RNNHidden<Weights1, Weights2>::RNNHidden(const Weights1& initModel, const Weights2& gruModel)
: w_(initModel), gru_(gruModel) {}

template <class Weights1, class Weights2>
void Decoder::RNNHidden<Weights1, Weights2>::InitializeState(mblas::Matrix& State,
                     const mblas::Matrix& SourceContext,
                     const size_t batchSize ) {
  using namespace mblas;

  // Calculate mean of source context, rowwise
  // Repeat mean batchSize times by broadcasting
  Temp1_ = Mean<byRow, Matrix>(SourceContext);
  Temp2_.resize(batchSize, SourceContext.columns());
  Temp2_ = 0.0f;
  AddBiasVector<byRow>(Temp2_, Temp1_);

  State = Temp2_ * w_.Wi_;
  AddBiasVector<byRow>(State, w_.Bi_);

  State = blaze::forEach(State, Tanh());
}

template <class Weights1, class Weights2>
void Decoder::RNNHidden<Weights1, Weights2>::GetNextState(mblas::Matrix& NextState,
                  const mblas::Matrix& State,
                  const mblas::Matrix& Context) {
  gru_.GetNextState(NextState, State, Context);
}

//////////////////////////////////////////////////////////////
template <class Weights>
Decoder::RNNFinal<Weights>::RNNFinal(const Weights& model)
: gru_(model) {}

template <class Weights>
void Decoder::RNNFinal<Weights>::GetNextState(mblas::Matrix& NextState,
                  const mblas::Matrix& State,
                  const mblas::Matrix& Context) {
  gru_.GetNextState(NextState, State, Context);
}

//////////////////////////////////////////////////////////////
Decoder::Decoder(const Weights& model)
: embeddings_(model.decEmbeddings_),
  rnn1_(model.decInit_, model.decGru1_),
  rnn2_(model.decGru2_),
  attention_(model.decAttention_),
  softmax_(model.decSoftmax_)
{}

void Decoder::MakeStep(mblas::Matrix& NextState,
			  mblas::ArrayMatrix& Probs,
			  const mblas::Matrix& State,
			  const mblas::Matrix& Embeddings,
			  const mblas::Matrix& SourceContext) {
  GetHiddenState(HiddenState_, State, Embeddings);
  GetAlignedSourceContext(AlignedSourceContext_, HiddenState_, SourceContext);
  GetNextState(NextState, HiddenState_, AlignedSourceContext_);
  GetProbs(Probs, NextState, Embeddings, AlignedSourceContext_);
}

void Decoder::EmptyState(mblas::Matrix& State,
				const mblas::Matrix& SourceContext,
				size_t batchSize) {
  rnn1_.InitializeState(State, SourceContext, batchSize);
}

void Decoder::EmptyEmbedding(mblas::Matrix& Embedding,
					size_t batchSize) {
  Embedding.resize(batchSize, embeddings_.GetCols());
  Embedding = 0.0f;
}

void Decoder::Lookup(mblas::Matrix& Embedding,
			const std::vector<size_t>& w) {
  embeddings_.Lookup(Embedding, w);
}

void Decoder::Filter(const std::vector<size_t>& ids) {
  softmax_.Filter(ids);
}

void Decoder::GetAttention(mblas::Matrix& attention) {
	attention_.GetAttention(attention);
}

size_t Decoder::GetVocabSize() const {
  return embeddings_.GetRows();
}

void Decoder::GetHiddenState(mblas::Matrix& HiddenState,
                    const mblas::Matrix& PrevState,
                    const mblas::Matrix& Embedding) {
  rnn1_.GetNextState(HiddenState, PrevState, Embedding);
}

void Decoder::GetAlignedSourceContext(mblas::Matrix& AlignedSourceContext,
                             const mblas::Matrix& HiddenState,
                             const mblas::Matrix& SourceContext) {
	attention_.GetAlignedSourceContext(AlignedSourceContext, HiddenState, SourceContext);
}

void Decoder::GetNextState(mblas::Matrix& State,
                  const mblas::Matrix& HiddenState,
                  const mblas::Matrix& AlignedSourceContext) {
  rnn2_.GetNextState(State, HiddenState, AlignedSourceContext);
}


void Decoder::GetProbs(mblas::ArrayMatrix& Probs,
              const mblas::Matrix& State,
              const mblas::Matrix& Embedding,
              const mblas::Matrix& AlignedSourceContext) {
  softmax_.GetProbs(Probs, State, Embedding, AlignedSourceContext);
}
