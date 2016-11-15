#include "encoder.h"

using namespace std;

namespace GPU {

Encoder::Encoder(const Weights& model)
: embeddings_(model.encEmbeddings_),
  forwardRnn_(model.encForwardGRU_),
  backwardRnn_(model.encBackwardGRU_)
{}

size_t GetMaxLength(const Sentences& source, size_t tab) {
  size_t maxLength = source[0].GetWords(tab).size();
  for (const auto& sentence : source) {
    maxLength = std::max(maxLength, sentence.GetWords(tab).size());
  }
  return maxLength;
}

std::vector<std::vector<size_t>> GetBatchInput(const Sentences& source, size_t tab, size_t maxLen) {
  std::vector<std::vector<size_t>> matrix(maxLen, std::vector<size_t>(source.size(), 0));

  for (size_t i = 0; i < maxLen; ++i) {
    for (size_t j = 0; j < source.size(); ++j) {
      matrix[i][j] = source[j].GetWords(tab)[i];

    }
  }

  return matrix;
}

void Encoder::GetContext(const Sentences& source, size_t tab, mblas::Matrix& Context,
                         DeviceVector<int>& dMapping) {
  size_t maxSentenceLength = GetMaxLength(source, tab);

  thrust::host_vector<int> hMapping(maxSentenceLength * source.size(), 0);
  for (size_t i = 0; i < source.size(); ++i) {
    for (size_t j = 0; j < source[i].GetWords(tab).size(); ++j) {
      hMapping[i * maxSentenceLength + j] = 1;
    }
  }

  dMapping = hMapping;

  Context.Resize(maxSentenceLength * source.size(),
                 forwardRnn_.GetStateLength() + backwardRnn_.GetStateLength());

  auto input = GetBatchInput(source, tab, maxSentenceLength);

  for (size_t i = 0; i < input.size(); ++i) {
    if (i >= embeddedWords_.size()) {
      embeddedWords_.emplace_back();
    }
    embeddings_.Lookup(embeddedWords_[i], input[i]);
  }

  forwardRnn_.GetContext(embeddedWords_.cbegin(),
                         embeddedWords_.cbegin() + maxSentenceLength,
                         Context, source.size(), false);

  backwardRnn_.GetContext(embeddedWords_.crend() - maxSentenceLength,
                          embeddedWords_.crend(),
                          Context, source.size(), true);
}

}

