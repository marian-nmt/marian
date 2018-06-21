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

unsigned GetMaxLength(const Sentences& source, unsigned tab) {
  unsigned maxLength = source.Get(0).GetWords(tab).size();
  for (unsigned i = 0; i < source.size(); ++i) {
    const Sentence &sentence = source.Get(i);
    maxLength = std::max(maxLength, (unsigned) sentence.GetWords(tab).size());
  }
  return maxLength;
}

std::vector<std::vector<FactWord>> GetBatchInput(const Sentences& source, unsigned tab, unsigned maxLen) {
  std::vector<std::vector<FactWord>> matrix(maxLen, std::vector<FactWord>(source.size()));

  for (unsigned batchIdx = 0; batchIdx < source.size(); ++batchIdx) {
    for (unsigned wordIdx = 0; wordIdx < source.Get(batchIdx).GetFactors(tab).size(); ++wordIdx) {
        matrix[wordIdx][batchIdx] = source.Get(batchIdx).GetFactors(tab)[wordIdx];
    }
  }

  return matrix;
}

void Encoder::Encode(const Sentences& source,
                      unsigned tab,
                      mblas::Tensor& context,
                      std::vector<unsigned> &h_sentenceLengths,
                      mblas::Vector<unsigned> &sentenceLengths)
{
  unsigned maxSentenceLength = GetMaxLength(source, tab);

  h_sentenceLengths.resize(source.size());
  sentenceLengths.newSize(source.size());

  for (unsigned i = 0; i < source.size(); ++i) {
    h_sentenceLengths[i] = source.Get(i).GetWords(tab).size();
  }

  mblas::copy(h_sentenceLengths.data(),
              h_sentenceLengths.size(),
              sentenceLengths.data(),
              cudaMemcpyHostToDevice);

  //cerr << "GetContext1=" << context.Debug(1) << endl;
  context.NewSize(maxSentenceLength,
                 forwardRnn_.GetStateLength().output + backwardRnn_.GetStateLength().output,
                 1,
                 source.size());
  //cerr << "GetContext2=" << context.Debug(1) << endl;

  auto input = GetBatchInput(source, tab, maxSentenceLength);

  for (unsigned i = 0; i < input.size(); ++i) {
    if (i >= embeddedWords_.size()) {
      embeddedWords_.emplace_back();
    }
    embeddings_.Lookup(embeddedWords_[i], input[i]);
    //cerr << "embeddedWord_=" << embeddedWords_[i].Debug(1) << endl;
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

