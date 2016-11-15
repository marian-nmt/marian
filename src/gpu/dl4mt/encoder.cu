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

  /* for (size_t i = 0; i < maxLen; ++i) { */
    /* for (size_t j = 0; j < source.size(); ++j) { */
      /* std::cerr << matrix[i][j] << " "; */

    /* } */
    /* std::cerr << std::endl; */
  /* } */

  return matrix;
}

void Encoder::GetContext(const Sentences& source, size_t tab, mblas::Matrix& Context) {
  size_t maxSentenceLength = GetMaxLength(source, tab);
  /* std::cerr << ">> " << "Starting getting context. " << maxSentenceLength << std::endl; */
  /* std::cerr << maxSentenceLength * source.size() << " x " << */
                 /* forwardRnn_.GetStateLength() + backwardRnn_.GetStateLength() << std::endl; */
  Context.Resize(maxSentenceLength * source.size(),
                 forwardRnn_.GetStateLength() + backwardRnn_.GetStateLength());

  /* std::cerr << ">> " << "Starting getting batch input (transpose)." << std::endl; */
  auto input = GetBatchInput(source, tab, maxSentenceLength);

  /* std::cerr << ">> " << "Starting looking up." << std::endl; */
  for (size_t i = 0; i < input.size(); ++i) {
    if (i >= embeddedWords_.size()) {
      embeddedWords_.emplace_back();
    }
    embeddings_.Lookup(embeddedWords_[i], input[i]);
  }

  /* std::cerr << ">> " << "Starting forward." << std::endl; */
  forwardRnn_.GetContext(embeddedWords_.cbegin(),
                         embeddedWords_.cbegin() + maxSentenceLength,
                         Context, source.size(), false);

  /* std::cerr << ">> " << "Starting backward." << std::endl; */
  backwardRnn_.GetContext(embeddedWords_.crend() - maxSentenceLength,
                          embeddedWords_.crend(),
                          Context, source.size(), true);
  /* std::cerr << ">> " << "Finished getcontext." << std::endl; */
}

}

