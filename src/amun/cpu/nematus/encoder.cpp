#include "encoder.h"

using namespace std;

namespace amunmt {
namespace CPU {
namespace Nematus {

void Encoder::GetContext(const std::vector<unsigned>& words, mblas::Tensor& context) {
  std::vector<mblas::Tensor> embeddedWords;

  context.resize(words.size(),
                 forwardRnn_.GetStateLength() + backwardRnn_.GetStateLength());

  for (auto& w : words) {
    embeddedWords.emplace_back();
    mblas::Tensor &embed = embeddedWords.back();
    embeddings_.Lookup(embed, w);
  }

  forwardRnn_.GetContext(embeddedWords.cbegin(),
						 embeddedWords.cend(),
						 context, false);
  backwardRnn_.GetContext(embeddedWords.crbegin(),
						  embeddedWords.crend(),
						  context, true);
}

}  // namespace Nematus
}  // namespace CPU
}  // namespace amunmt

