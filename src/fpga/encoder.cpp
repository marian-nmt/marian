#include "encoder.h"

using namespace std;

namespace amunmt {
namespace FPGA {

Encoder::Encoder(const cl_context &context, const cl_device_id &device, const Weights& model)
: embeddings_(model.encEmbeddings_)
, forwardRnn_(context, device, model.encForwardGRU_)
, backwardRnn_(context, device, model.encBackwardGRU_)
, Context(context, device)
, context_(context)
, device_(device)
{

}

size_t GetMaxLength(const Sentences& source, size_t tab) {
  size_t maxLength = source.at(0)->GetWords(tab).size();
  for (size_t i = 0; i < source.size(); ++i) {
    const Sentence &sentence = *source.at(i);
    maxLength = std::max(maxLength, sentence.GetWords(tab).size());
  }
  return maxLength;
}


std::vector<std::vector<size_t>> GetBatchInput(const Sentences& source, size_t tab, size_t maxLen) {
  std::vector<std::vector<size_t>> matrix(maxLen, std::vector<size_t>(source.size(), 0));

  for (size_t j = 0; j < source.size(); ++j) {
    for (size_t i = 0; i < source.at(j)->GetWords(tab).size(); ++i) {
        matrix[i][j] = source.at(j)->GetWords(tab)[i];
    }
  }

  return matrix;
}

void Encoder::GetContext(const Sentences& source, size_t tab, mblas::Matrix& Context)
{
  size_t maxSentenceLength = GetMaxLength(source, tab);

  Context.Resize(maxSentenceLength,
                 forwardRnn_.GetStateLength() + backwardRnn_.GetStateLength(),
                 1,
                 source.size());

  auto input = GetBatchInput(source, tab, maxSentenceLength);

  for (size_t i = 0; i < input.size(); ++i) {
    if (i >= embeddedWords_.size()) {
      embeddedWords_.emplace_back(context_, device_);
    }
    embeddings_.Lookup(context_, device_, embeddedWords_[i], input[i]);
    cerr << "embeddedWords_=" << embeddedWords_.back().Debug(true) << endl;
  }

  forwardRnn_.GetContext(embeddedWords_.cbegin(),
                         embeddedWords_.cbegin() + maxSentenceLength,
                         Context, source.size(), false);

  backwardRnn_.GetContext(embeddedWords_.crend() - maxSentenceLength,
                          embeddedWords_.crend() ,
                          Context, source.size(), true);

}

}
}
