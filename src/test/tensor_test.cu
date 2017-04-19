#include <iostream>
#include <map>
#include <boost/timer/timer.hpp>

#include "data/corpus.h"
#include "training/config.h"
#include "tensors/tensor_allocator.h"

#include "graph/expression_operators.h"
#include "common/logging.h"

#include "models/lex_probs.h"

int main(int argc, char** argv) {
  using namespace marian;
  
  marian::Config config(argc, argv);
  
  auto srcVocab = New<Vocab>();
  auto trgVocab = New<Vocab>();
  
  srcVocab->load("model/vocab.ro.yml");
  trgVocab->load("model/vocab.en.yml");
  
  int srcDim = 50;
  int trgDim = 50;

  auto probs = New<LexProbs>("data/lex.s2t",
                             srcVocab, trgVocab,
                             srcDim, trgDim, 0);
  
  TensorAllocator ta(0);
  
  int batchSize = 1;
  int srcWords = 6;
  int trgWords = 2;

  
  std::vector<Ptr<data::SubBatch>> batches;
  batches.push_back(New<data::SubBatch>(batchSize, srcWords));
  batches.back()->indeces() = { 3, 4, 0, 1, 2, 0 };
  
  auto batch = New<data::CorpusBatch>(batches);
  
  Tensor att, logits;
  Tensor lf, lfa;
  
  ta.allocate(att, {batchSize, 1, srcWords, trgWords});
  ta.allocate(logits, {batchSize, trgDim, trgWords});
  ta.allocate(lf, {batchSize, trgDim, srcWords, 1});
  ta.allocate(lfa, {batchSize, trgDim, trgWords});
  
  logits->set(0);

  auto slf = probs->Lf(batch);
  slf->toTensor(lf);
  std::cerr << lf->debug() << std::endl;
  
  std::vector<float> av = { 0.9, 0.05, 0.02, 0.01, 0.01, 0.01,
                            0.9, 0.05, 0.02, 0.01, 0.01, 0.01 };
  att->set(av);
  std::cerr << att->debug() << std::endl;
  
  sparse::LfaForward(lfa, logits, att, slf);
  std::cerr << lfa->debug() << std::endl;
  
  
  return 0;
}
