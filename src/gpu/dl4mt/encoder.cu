#include "encoder.h"

using namespace std;

namespace GPU {

Encoder::Encoder(const Weights& model)
: embeddings_(model.encEmbeddings_),
  forwardRnn_(model.encForwardGRU_),
  backwardRnn_(model.encBackwardGRU_)
{}

void Encoder::GetContext(const std::vector<size_t>& words, mblas::Matrix& Context) {
  Context.Resize(words.size(), forwardRnn_.GetStateLength()
                               + backwardRnn_.GetStateLength());
  for (size_t i = 0; i < words.size(); ++i) {
    if (i >= embeddedWords_.size()) {
      embeddedWords_.emplace_back();
    }
    embeddings_.Lookup(embeddedWords_[i], words[i]);
  }

  forwardRnn_.GetContext(embeddedWords_.cbegin(),
                         embeddedWords_.cbegin() + words.size(),
                         Context, false);

  backwardRnn_.GetContext(embeddedWords_.crend() - words.size(),
                          embeddedWords_.crend(),
                          Context, true);
}

}

