#include "encoder.h"

using namespace std;

namespace amunmt {
namespace CPU {
namespace dl4mt {

void Encoder::Encode(const std::vector<uint>& words,
				mblas::Matrix& context) {
  std::vector<mblas::Matrix> embeddedWords;

  context.resize(words.size(),
				 forwardRnn_.GetStateLength()
				 + backwardRnn_.GetStateLength());
  for(auto& w : words) {
    embeddedWords.emplace_back();
    mblas::Matrix &embed = embeddedWords.back();
    embeddings_.Lookup(embed, w);
    //cerr << "embed=" << embed.Debug(true) << endl;
  }

  forwardRnn_.Encode(embeddedWords.cbegin(),
						 embeddedWords.cend(),
						 context, false);
  backwardRnn_.Encode(embeddedWords.crbegin(),
						  embeddedWords.crend(),
						  context, true);
}

}
}
}

