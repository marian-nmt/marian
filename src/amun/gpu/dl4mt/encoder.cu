#include "encoder.h"
#include "common/sentences.h"

using namespace std;

namespace amunmt {
namespace GPU {

Encoder::Encoder(const Weights& model, const YAML::Node& config)
  : embeddings_(model.encEmbeddings_),
    forwardRnn_(InitForwardCell(model, config)),
    backwardRnn_(InitBackwardCell(model, config))
{}

std::unique_ptr<Cell> Encoder::InitForwardCell(const Weights& model, const YAML::Node& config){
  std::string celltype = config["enc-cell"] ? config["enc-cell"].as<std::string>() : "gru";
  if (celltype == "lstm") {
    return unique_ptr<Cell>(new LSTM<Weights::EncForwardLSTM>(*(model.encForwardLSTM_)));
  } else if (celltype == "mlstm") {
    return unique_ptr<Cell>(new Multiplicative<LSTM, Weights::EncForwardLSTM>(*model.encForwardMLSTM_));
  } else if (celltype == "gru") {
    return unique_ptr<Cell>(new GRU<Weights::EncForwardGRU>(*(model.encForwardGRU_)));
  }

  assert(false);
  return unique_ptr<Cell>(nullptr);
}

std::unique_ptr<Cell> Encoder::InitBackwardCell(const Weights& model, const YAML::Node& config){
  std::string enccell = config["enc-cell"] ? config["enc-cell"].as<std::string>() : "gru";
  std::string celltype = config["enc-cell-r"] ? config["enc-cell-r"].as<std::string>() : enccell;
  if (celltype == "lstm") {
    return unique_ptr<Cell>(new LSTM<Weights::EncBackwardLSTM>(*(model.encBackwardLSTM_)));
  } else if (celltype == "mlstm") {
    return unique_ptr<Cell>(new Multiplicative<LSTM, Weights::EncBackwardLSTM>(*model.encBackwardMLSTM_));
  } else if (celltype == "gru") {
    return unique_ptr<Cell>(new GRU<Weights::EncBackwardGRU>(*(model.encBackwardGRU_)));
  }

  assert(false);
  return unique_ptr<Cell>(nullptr);
}

size_t GetMaxLength(const Sentences& source, size_t tab) {
  size_t maxLength = source.at(0)->GetWords(tab).size();
  for (size_t i = 0; i < source.size(); ++i) {
    const Sentence &sentence = *source.at(i);
    maxLength = std::max(maxLength, sentence.GetWords(tab).size());
  }
  return maxLength;
}

std::vector<std::vector<uint>> GetBatchInput(const Sentences& source, size_t tab, size_t maxLen) {
  std::vector<std::vector<uint>> matrix(maxLen, std::vector<uint>(source.size(), 0));

  for (size_t j = 0; j < source.size(); ++j) {
    for (size_t i = 0; i < source.at(j)->GetWords(tab).size(); ++i) {
        matrix[i][j] = source.at(j)->GetWords(tab)[i];
    }
  }

  return matrix;
}

void Encoder::Encode(const Sentences& source, size_t tab, mblas::Matrix& context,
                         mblas::IMatrix &sentenceLengths)
{
  size_t maxSentenceLength = GetMaxLength(source, tab);

  std::vector<uint> hSentenceLengths(source.size());
  for (size_t i = 0; i < source.size(); ++i) {
    hSentenceLengths[i] = source.at(i)->GetWords(tab).size();
  }

  sentenceLengths.NewSize(source.size(), 1, 1, 1);
  mblas::copy(hSentenceLengths.data(),
              hSentenceLengths.size(),
              sentenceLengths.data(),
              cudaMemcpyHostToDevice);

  //cerr << "GetContext1=" << context.Debug(1) << endl;
  context.NewSize(maxSentenceLength,
                 forwardRnn_.GetStateLength().output + backwardRnn_.GetStateLength().output,
                 1,
                 source.size());
  //cerr << "GetContext2=" << context.Debug(1) << endl;

  auto input = GetBatchInput(source, tab, maxSentenceLength);

  for (size_t i = 0; i < input.size(); ++i) {
    if (i >= embeddedWords_.size()) {
      embeddedWords_.emplace_back();
    }
    embeddings_.Lookup(embeddedWords_[i], input[i]);
    //cerr << "embeddedWords_=" << embeddedWords_.back().Debug(true) << endl;
  }

  //cerr << "GetContext3=" << context.Debug(1) << endl;
  forwardRnn_.Encode(embeddedWords_.cbegin(),
                         embeddedWords_.cbegin() + maxSentenceLength,
                         context, source.size(), false);
  //cerr << "GetContext4=" << context.Debug(1) << endl;

  backwardRnn_.Encode(embeddedWords_.crend() - maxSentenceLength,
                          embeddedWords_.crend() ,
                          context, source.size(), true, &sentenceLengths);
  //cerr << "GetContext5=" << context.Debug(1) << endl;
}

}
}

