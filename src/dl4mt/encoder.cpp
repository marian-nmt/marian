#include "encoder.h"

using namespace std;

void Encoder::GetContext(const std::vector<size_t>& words,
				mblas::Matrix& context) {
  std::vector<mblas::Matrix> embeddedWords;

  context.Resize(words.size(), forwardRnn_.GetStateLength() + backwardRnn_.GetStateLength());
  for(auto& w : words) {
	embeddedWords.emplace_back();
	mblas::Matrix &embed = embeddedWords.back();
	embeddings_.Lookup(embed, w);

	//cerr << "w=" << w << embed << endl;
  }

  forwardRnn_.GetContext(embeddedWords.cbegin(),
						 embeddedWords.cend(),
						 context, false);
  backwardRnn_.GetContext(embeddedWords.crbegin(),
						  embeddedWords.crend(),
						  context, true);
}
