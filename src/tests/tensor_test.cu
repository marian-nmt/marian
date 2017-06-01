#include <iostream>
#include <map>
#include <boost/timer/timer.hpp>

#include "training/config.h"
#include "tensors/tensor_allocator.h"

#include "kernels/tensor_operators.h"
#include "common/logging.h"

int main(int argc, char** argv) {
  using namespace marian;
  
  marian::Config config(argc, argv);
  
  TensorAllocator ta(0);
  
  int batchSize = 64;
  int hidden = 2048;
  int words = 13;
  
  Tensor out1, out2, out3;
  Tensor a_1, a_2, a_3;
  Tensor b_1, b_2, b_3;
  
  ta.allocate(out1, {batchSize, hidden, 1});
  ta.allocate(out2, {batchSize, hidden, 1});
  ta.allocate(out3, {batchSize, hidden, words});
  
  ta.allocate(a1, {batchSize, hidden, 1});
  ta.allocate(a2, {batchSize, hidden, 1});
  ta.allocate(a3, {batchSize, 1, words});

  ta.allocate(b1, {1, hidden, 1});
  ta.allocate(b2, {batchSize, 1, 1});
  ta.allocate(b3, {batchSize, hidden, 1});
  
  out1->set(0);
  out2->set(0);
  out3->set(0);
  
  a1->set(1);
  a2->set(1);
  a3->set(1);
  
  b1->set(2);
  b2->set(2);
  b3->set(2);
  
  for(int i = 0; i < 100; i++) {
    Add(_1 * _2, out1, a1, b1);
    Add(_1 * _2, out2, a2, b2);
    Add(_1 * _2, out3, a3, b3);
  }
  
  //auto srcVocab = New<Vocab>();
  //auto trgVocab = New<Vocab>();
  //
  //srcVocab->load("model/vocab.ro.yml");
  //trgVocab->load("model/vocab.en.yml");
  //
  //int srcDim = 50;
  //int trgDim = 50;
  //
  //auto probs = New<LexProbs>("data/lex.s2t",
  //                           srcVocab, trgVocab,
  //                           srcDim, trgDim, 0);
  //
  //TensorAllocator ta(0);
  //
  //int batchSize = 1;
  //int srcWords = 6;
  //int trgWords = 2;
  //
  //
  //std::vector<Ptr<data::SubBatch>> batches;
  //batches.push_back(New<data::SubBatch>(batchSize, srcWords));
  //batches.back()->indeces() = { 3, 4, 0, 1, 2, 0 };
  //
  //auto batch = New<data::CorpusBatch>(batches);
  //
  //Tensor att, logits;
  //Tensor lf, lfa;
  //
  //ta.allocate(att, {batchSize, 1, srcWords, trgWords});
  //ta.allocate(logits, {batchSize, trgDim, trgWords});
  //ta.allocate(lf, {batchSize, trgDim, srcWords, 1});
  //ta.allocate(lfa, {batchSize, trgDim, trgWords});
  //
  //logits->set(0);
  //
  //auto slf = probs->Lf(batch);
  //slf->toTensor(lf);
  //std::cerr << lf->debug() << std::endl;
  //
  //std::vector<float> av = { 0.9, 0.05, 0.02, 0.01, 0.01, 0.01,
  //                          0.9, 0.05, 0.02, 0.01, 0.01, 0.01 };
  //att->set(av);
  //std::cerr << att->debug() << std::endl;
  //
  //sparse::LfaForward(lfa, logits, att, slf);
  //std::cerr << lfa->debug() << std::endl;
  
  
  return 0;
}
