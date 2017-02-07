#include "encoder.h"

using namespace std;

namespace amunmt {
namespace CPU {

void Encoder::GetContext(const std::vector<size_t>& words,
				mblas::Matrix& context) {
  std::vector<mblas::Matrix> embeddedWords;

  context.resize(words.size(),
				 forwardRnn_.GetStateLength()
				 + backwardRnn_.GetStateLength());
  for(auto& w : words) {
	embeddedWords.emplace_back();
	mblas::Matrix &embed = embeddedWords.back();
	embeddings_.Lookup(embed, w);
  }
  //cerr << embeddings_.w_.E_.Debug() << endl;

  forwardRnn_.GetContext(embeddedWords.cbegin(),
						 embeddedWords.cend(),
						 context, false);
  backwardRnn_.GetContext(embeddedWords.crbegin(),
						  embeddedWords.crend(),
						  context, true);
}

}
}

