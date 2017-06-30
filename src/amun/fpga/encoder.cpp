#include "encoder.h"

using namespace std;

namespace amunmt {
namespace FPGA {

Encoder::Encoder(const OpenCLInfo &openCLInfo, const Weights& model)
: embeddings_(model.encEmbeddings_)
, forwardRnn_(openCLInfo, model.encForwardGRU_)
, backwardRnn_(openCLInfo, model.encBackwardGRU_)
, Context(openCLInfo)
, openCLInfo_(openCLInfo)
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

void Encoder::Encode(const Sentences& source, size_t tab, mblas::Matrix& context,
                        Array<int>& dMapping)
{
  size_t maxSentenceLength = GetMaxLength(source, tab);

  std::vector<int> hMapping(maxSentenceLength * source.size(), 0);
  for (size_t i = 0; i < source.size(); ++i) {
    for (size_t j = 0; j < source.at(i)->GetWords(tab).size(); ++j) {
      hMapping[i * maxSentenceLength + j] = 1;
    }
  }

  Array<int> temp(dMapping.GetOpenCLInfo(), hMapping);
  temp.Swap(dMapping);

  //cerr << "dMapping=" << dMapping.Debug(2) << endl;

  //cerr << "GetContext1=" << context.Debug(1) << endl;
  context.Resize(maxSentenceLength,
                 forwardRnn_.GetStateLength() + backwardRnn_.GetStateLength(),
                 1,
                 source.size());
  //cerr << "GetContext2=" << context.Debug(1) << endl;

  auto input = GetBatchInput(source, tab, maxSentenceLength);

  for (size_t i = 0; i < input.size(); ++i) {
    if (i >= embeddedWords_.size()) {
      embeddedWords_.emplace_back(openCLInfo_);
    }
    embeddings_.Lookup(openCLInfo_, embeddedWords_[i], input[i]);
    //cerr << "embeddedWords_=" << embeddedWords_.back().Debug(1) << endl;
  }

  //cerr << "GetContext3=" << context.Debug(1) << endl;
  forwardRnn_.Encode(embeddedWords_.cbegin(),
                         embeddedWords_.cbegin() + maxSentenceLength,
                         context, source.size(), false);
  //cerr << "GetContext4=" << context.Debug(1) << endl;

  backwardRnn_.Encode(embeddedWords_.crend() - maxSentenceLength,
                          embeddedWords_.crend() ,
                          context, source.size(),
                          true,
                          &dMapping);
  //cerr << "GetContext5=" << context.Debug(1) << endl;

}

}
}
