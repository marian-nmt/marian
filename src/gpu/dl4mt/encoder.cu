#include "encoder.h"

Encoder::Encoder(const Weights& model)
: embeddings_(model.encEmbeddings_),
  forwardRnn_(model.encForwardGRU_),
  backwardRnn_(model.encBackwardGRU_)
{}

void Encoder::GetContext(const std::vector<size_t>& words,
				mblas::Matrix& Context) {
  std::vector<mblas::Matrix> embeddedWords;

  Context.Resize(words.size(), forwardRnn_.GetStateLength() + backwardRnn_.GetStateLength());
  for(auto& w : words) {
	embeddedWords.emplace_back();
	embeddings_.Lookup(embeddedWords.back(), w);
  }

  forwardRnn_.GetContext(embeddedWords.cbegin(),
						 embeddedWords.cend(),
						 Context, false);
  backwardRnn_.GetContext(embeddedWords.crbegin(),
						  embeddedWords.crend(),
						  Context, true);
}

