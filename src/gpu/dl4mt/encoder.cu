#include "encoder.h"

using namespace std;

namespace GPU {

Encoder::Encoder(const Weights& model)
: embeddings_(model.encEmbeddings_),
  forwardRnn_(model.encForwardGRU_),
  backwardRnn_(model.encBackwardGRU_)
{}

void Encoder::GetContext(const std::vector<size_t>& words,
				mblas::Matrix& Context) {
  thread_local static std::vector<mblas::Matrix> embeddedWords;

  Context.Resize(words.size(), forwardRnn_.GetStateLength() + backwardRnn_.GetStateLength());
  for (size_t i = 0; i < words.size(); ++i) {
    if (i >= embeddedWords.size()) {
      embeddedWords.emplace_back();
    }
    embeddings_.Lookup(embeddedWords[i], words[i]);
  }
  //cerr << "embeddings_=" << embeddings_.w_.E_.Debug() << endl;

  forwardRnn_.GetContext(embeddedWords.cbegin(),
						 embeddedWords.cbegin() + words.size(),
						 Context, false);
  backwardRnn_.GetContext(embeddedWords.crend() - words.size(),
						  embeddedWords.crend(),
						  Context, true);
}

}

