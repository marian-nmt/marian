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
                         mblas::IMatrix &sentencesMask)
{
  size_t maxSentenceLength = GetMaxLength(source, tab);
  size_t maxMergedLength = maxSentenceLength / embeddings_.FactorCount();

  //cerr << "1dMapping=" << mblas::Debug(dMapping, 2) << endl;
  HostVector<uint> hMapping(maxMergedLength * source.size(), 0);
  for (size_t i = 0; i < source.size(); ++i) {
    for (size_t j = 0; j < source.at(i)->GetWords(tab).size() / embeddings_.FactorCount(); ++j) {
      hMapping[i * maxMergedLength + j] = 1;
    }
  }

  sentencesMask.NewSize(maxMergedLength, source.size(), 1, 1);
  mblas::copy(thrust::raw_pointer_cast(hMapping.data()),
              hMapping.size(),
              sentencesMask.data(),
              cudaMemcpyHostToDevice);

  //cerr << "GetContext1=" << context.Debug(1) << endl;
  context.NewSize(maxMergedLength,
                 forwardRnn_.GetStateLength().output + backwardRnn_.GetStateLength().output,
                 1,
                 source.size());
  //cerr << "GetContext2=" << context.Debug(1) << endl;

  auto input = GetBatchInput(source, tab, maxSentenceLength);
  // input is a sentence; sentence is a vector of batches; batch is a vector of words
  // we'll convert each word into a vector of factors by combining every number-of-factors
  // batches together
  std::vector<std::vector<std::vector<size_t>>> mergedInput(input.size() / embeddings_.FactorCount());
  for (size_t i = 0; i < input.size(); ) {
    std::vector<std::vector<size_t>> newbatch
      // asume that batchsize is the same for each of the factors of a single word
      (input[i].size(), std::vector<size_t>(embeddings_.FactorCount()));

    for (size_t factorIdx = 0; factorIdx < embeddings_.FactorCount(); ++factorIdx) {
      const std::vector<size_t>& batch = input[i];

      for (size_t j = 0; j < batch.size(); ++j) {
        newbatch.at(j)[factorIdx] = batch[j];
      }
      ++i;
    }
    mergedInput[i / embeddings_.FactorCount() - 1] = newbatch;
  }

  for (size_t i = 0; i < mergedInput.size(); ++i) {
    if (i >= embeddedWords_.size()) {
      embeddedWords_.emplace_back();
    }
    embeddings_.Lookup(embeddedWords_[i], mergedInput[i]);
    //cerr << "embeddedWords_=" << embeddedWords_.back().Debug(true) << endl;
  }

  //cerr << "GetContext3=" << context.Debug(1) << endl;
  forwardRnn_.Encode(embeddedWords_.cbegin(),
                         embeddedWords_.cbegin() + maxMergedLength,
                         context, source.size(), false);
  //cerr << "GetContext4=" << context.Debug(1) << endl;

  backwardRnn_.Encode(embeddedWords_.crend() - maxMergedLength,
                          embeddedWords_.crend() ,
                          context, source.size(), true, &sentencesMask);
  //cerr << "GetContext5=" << context.Debug(1) << endl;
}

}
}

