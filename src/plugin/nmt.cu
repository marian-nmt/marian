#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <cmath>
#include <boost/timer/timer.hpp>

#include "nmt.h"
#include "common/vocab.h"
#include "common/god.h"
#include "common/history.h"

using namespace GPU;

void MosesPlugin::initGod(const std::string& configPath) {
  std::string configs = "-c " + configPath;
  god_ = new God();
  god_->Init(configs);
}

MosesPlugin::MosesPlugin()
  : debug_(false),
    states_(new States()),
    firstWord_(true),
    scorers_(god_->GetScorers(1)),
    bestHyps_(god_->GetBestHyps(1))
{}

size_t MosesPlugin::GetDevices(size_t maxDevices) {
  int num_gpus = 0; // number of CUDA GPUs
  cudaGetDeviceCount(&num_gpus);
  std::cerr << "Number of CUDA devices: " << num_gpus << std::endl;

  for (int i = 0; i < num_gpus; i++) {
      cudaDeviceProp dprop;
      cudaGetDeviceProperties(&dprop, i);
      std::cerr << i << ": " << dprop.name << std::endl;
  }
  return (size_t)std::min(num_gpus, (int)maxDevices);
}


void MosesPlugin::GeneratePhrases(const States& states, std::string& lastWord, size_t numPhrases,
                                  std::vector<NeuralPhrase>& phrases) {
  Histories histories(*god_, sentences_);

  size_t batchSize = 1;
  std::vector<size_t> beamSizes(batchSize, 1);

  Beam prevHyps(batchSize, HypothesisPtr(new Hypothesis()));
  for (size_t i = 0; i < histories.size(); ++i) {
    histories.at(i)->Add(prevHyps);
  }

  States nextStates(scorers_.size());

  size_t vocabSize = scorers_[0]->GetVocabSize();

  size_t maxLength = 5;

  for (size_t decoderStep = 0; decoderStep < maxLength; ++decoderStep) {
    for (size_t i = 0; i < scorers_.size(); i++) {
      Scorer &scorer = *scorers_[i];
      State &state = *states[i];
      State &nextState = *nextStates[i];

      scorer.Score(*god_, state, nextState, beamSizes);
    }

    if (decoderStep == 0) {
      for (auto& beamSize : beamSizes) {
        beamSize = god_->Get<size_t>("beam-size");
      }
    }

    Beams beams(batchSize);

    bestHyps_(*god_, beams, prevHyps, beamSizes, scorers_, filterIndices_, true);

    for (size_t i = 0; i < batchSize; ++i) {
      if (!beams[i].empty()) {
        histories.at(i)->Add(beams[i], histories.at(i)->size() == maxLength);
      }
    }

    Beam survivors;
    for (size_t batchID = 0; batchID < batchSize; ++batchID) {
      for (auto& h : beams[batchID]) {
        if (h->GetWord() != EOS) {
          survivors.push_back(h);
        } else {
          --beamSizes[batchID];
        }
      }
    }

    if (survivors.size() == 0) {
      break;
    }

    for (size_t i = 0; i < scorers_.size(); i++) {
      scorers_[i]->AssembleBeamState(*nextStates[i], survivors, *states[i]);
    }

    prevHyps.swap(survivors);
  }

  for (auto scorer : scorers_) {
	  scorer->CleanUpAfterSentence();
  }

  const NBestList &nbl = histories.at(0)->NBest(god_->Get<size_t>("beam-size"));

  for (size_t i = 0; i < nbl.size(); ++i) {
    const Result& result = nbl[i];
    auto words = god_->Postprocess(god_->GetTargetVocab()(result.first));
    auto& scores = result.second->GetCostBreakdown();

    phrases.emplace_back(words, scores, 0, 1);
  }
}
/* void MosesPlugin::SetDevice() { */
  /* cudaSetDevice(w_->GetDevice()); */
  /* CublasHandler::StaticHandle(); */
/* } */

/* size_t MosesPlugin::GetDevice() { */
  /* return w_->GetDevice(); */
/* } */

/* void MosesPlugin::ClearStates() { */
  /* firstWord_ = true; */
  /* states_->Clear(); */
/* } */

/* size_t MosesPlugin::TargetVocab(const std::string& str) { */
  /* return (*trg_)[str]; */
/* } */

States MosesPlugin::SetSource(const std::vector<std::string>& words) {
  if (sentences_.size() == 0) {
    sentences_.push_back(boost::shared_ptr<Sentence>(new Sentence(*god_, 0, words)));
  } else {
    sentences_.at(0).reset(new Sentence(*god_, 0, words));
  }

  States states(scorers_.size());

  for (size_t i = 0; i < scorers_.size(); ++i) {
    scorers_[i]->SetSource(sentences_);
    scorers_[i]->BeginSentenceState(*states[i], sentences_.size());
  }
  return states;
}

/* StateInfoPtr MosesPlugin::EmptyState() { */
  /* mblas::Matrix& SC = *boost::static_pointer_cast<mblas::Matrix>(SourceContext_); */
  /* mblas::Matrix Empty; */
  /* decoder_->EmptyState(Empty, SC, 1); */
  /* std::vector<StateInfoPtr> infos; */
  /* states_->SaveStates(infos, Empty); */
  /* return infos.back(); */
/* } */

/* void MosesPlugin::FilterTargetVocab(const std::set<std::string>& filter, size_t topN) { */
  /* filteredId_.resize(topN); */
  /* std::set<size_t> ids(filteredId_.begin(), filteredId_.end()); */
  /* filteredId_.resize(trg_->size(), 1); // set all to UNK */

  /* size_t k = topN; */
  /* for(auto& s : filter) { */
    /* size_t id = (*trg_)[s]; */
    /* if(ids.count(id) == 0) { */
      /* ids.insert(id); */
      /* filteredId_[id] = k; */
      /* k++; */
    /* } */
  /* } */
  /* // eol */
  /* std::vector<size_t> numericFilter(ids.begin(), ids.end()); */
  /* decoder_->Filter(numericFilter); */
/* } */

/* void MosesPlugin::BatchSteps(const Batches& batches, LastWords& lastWords, */
                     /* Scores& probsOut, Scores& unksOut, StateInfos& stateInfos, */
                     /* bool firstWord) { */
  /* mblas::Matrix& sourceContext = *boost::static_pointer_cast<mblas::Matrix>(SourceContext_); */

  /* mblas::Matrix prevEmbeddings; */
  /* mblas::Matrix nextEmbeddings; */
  /* mblas::Matrix prevStates; */
  /* mblas::Matrix probs; */
  /* mblas::Matrix nextStates; */

  /* if(firstWord) { */
    /* decoder_->EmptyEmbedding(prevEmbeddings, lastWords.size()); */
  /* } */
  /* else { */
    /* decoder_->Lookup(prevEmbeddings, lastWords); */
  /* } */

  /* states_->ConstructStates(prevStates, stateInfos); */

  /* for(auto& batch : batches) { */
    /* decoder_->MakeStep(nextStates, probs, prevStates, prevEmbeddings, sourceContext); */
    /* decoder_->Lookup(nextEmbeddings, batch); */
    /* StateInfos tempStates; */
    /* states_->SaveStates(tempStates, nextStates); */

    /* for(size_t i = 0; i < batch.size(); ++i) { */
      /* if(batch[i] != 0) { */
        /* float p = probs(i, filteredId_[batch[i]]); */
        /* probsOut[i] += log(p); */
        /* stateInfos[i] = tempStates[i]; */
      /* } */
      /* if(batch[i] == 1) { */
        /* unksOut[i]++; */
      /* } */
    /* } */

    /* Swap(nextStates, prevStates); */
    /* Swap(nextEmbeddings, prevEmbeddings); */
  /* } */
/* } */

/* void MosesPlugin::OnePhrase( */
  /* const std::vector<std::string>& phrase, */
  /* const std::string& lastWord, */
  /* bool firstWord, */
  /* StateInfoPtr inputState, */
  /* float& prob, size_t& unks, */
  /* StateInfoPtr& outputState) { */

  /* mblas::Matrix& sourceContext = *boost::static_pointer_cast<mblas::Matrix>(SourceContext_); */

  /* mblas::Matrix prevEmbeddings; */
  /* mblas::Matrix nextEmbeddings; */
  /* mblas::Matrix prevStates; */
  /* mblas::Matrix probs; */
  /* mblas::Matrix alignedSourceContext; */
  /* mblas::Matrix nextStates; */

  /* if(firstWord) { */
    /* decoder_->EmptyEmbedding(prevEmbeddings, 1); */
  /* } */
  /* else { */
    /* // Not the first word */
    /* std::vector<size_t> ids = { (*trg_)[lastWord] }; */
    /* decoder_->Lookup(prevEmbeddings, ids); */
  /* } */

  /* std::vector<StateInfoPtr> inputStates = { inputState }; */
  /* states_->ConstructStates(prevStates, inputStates); */

  /* for(auto& w : phrase) { */
    /* size_t id = (*trg_)[w]; */
    /* std::vector<size_t> nextIds = { id }; */
    /* if(id == 1) */
      /* unks++; */

    /* decoder_->MakeStep(nextStates, probs, */
                       /* prevStates, prevEmbeddings, sourceContext); */
    /* decoder_->Lookup(nextEmbeddings, nextIds); */
    /* float p = probs(0, filteredId_[id]); */
    /* prob += log(p); */

    /* Swap(nextStates, prevStates); */
    /* Swap(nextEmbeddings, prevEmbeddings); */
  /* } */

  /* std::vector<StateInfoPtr> outputStates; */
  /* states_->SaveStates(outputStates, prevStates); */
  /* outputState = outputStates.back(); */
/* } */

/* void MosesPlugin::MakeStep( */
  /* const std::vector<std::string>& nextWords, */
  /* const std::vector<std::string>& lastWords, */
  /* std::vector<StateInfoPtr>& inputStates, */
  /* std::vector<double>& logProbs, */
  /* std::vector<StateInfoPtr>& outputStates, */
  /* std::vector<bool>& unks) { */

  /* mblas::Matrix& sourceContext = *boost::static_pointer_cast<mblas::Matrix>(SourceContext_); */

  /* mblas::Matrix lastEmbeddings; */
  /* if(firstWord_) { */
    /* firstWord_ = false; */
    /* // Only empty state in state cache, so this is the first word */
    /* decoder_->EmptyEmbedding(lastEmbeddings, lastWords.size()); */
  /* } */
  /* else { */
    /* // Not the first word */
    /* std::vector<size_t> lastIds(lastWords.size()); */
    /* std::transform(lastWords.begin(), lastWords.end(), lastIds.begin(), */
                   /* [&](const std::string& w) { return (*trg_)[w]; }); */
    /* decoder_->Lookup(lastEmbeddings, lastIds); */
  /* } */

  /* mblas::Matrix nextEmbeddings; */
  /* std::vector<size_t> nextIds(nextWords.size()); */
  /* std::transform(nextWords.begin(), nextWords.end(), nextIds.begin(), */
                 /* [&](const std::string& w) { return (*trg_)[w]; }); */

  /* mblas::Matrix prevStates; */
  /* states_->ConstructStates(prevStates, inputStates); */

  /* mblas::Matrix probs; */
  /* mblas::Matrix nextStates; */

  /* decoder_->MakeStep(nextStates, probs, */
                     /* prevStates, lastEmbeddings, sourceContext); */
  /* decoder_->Lookup(nextEmbeddings, nextIds); */
  /* states_->SaveStates(outputStates, nextStates); */

  /* for(auto id : nextIds) { */
    /* if(id != 1) */
      /* unks.push_back(true); */
    /* else */
      /* unks.push_back(false); */
  /* } */

  /* for(size_t i = 0; i < nextIds.size(); ++i) { */
    /* float p = probs(i, filteredId_[nextIds[i]]); */
    /* //float p = probs(i, nextIds[i]); */
    /* logProbs.push_back(log(p)); */
  /* } */
/* } */

/* std::vector<double> MosesPlugin::RescoreNBestList( */
    /* const std::vector<std::string>& nbest, */
    /* const size_t maxBatchSize) { */

  /* mblas::Matrix PrevState; */
  /* mblas::Matrix PrevEmbedding; */
  /* mblas::Matrix Probs; */
  /* mblas::Matrix State; */
  /* mblas::Matrix Embedding; */

  /* NBest nBest(src_, trg_, nbest); */

  /* std::vector<double> nBestScores; */
  /* for (auto& batch: nBest.DivideNBestListIntoBatches()) { */
    /* size_t batchSize = batch[0].size(); */

    /* decoder_->EmptyState( */
        /* PrevState, */
        /* *boost::static_pointer_cast<mblas::Matrix>(SourceContext_), */
        /* batchSize); */
    /* decoder_->EmptyEmbedding(PrevEmbedding, batchSize); */

    /* std::vector<float> scores(batch[0].size(), 0.0f); */
    /* size_t lengthIndex = 0; */
    /* for (auto& w : batch) { */
      /* decoder_->MakeStep(State, Probs, PrevState, PrevEmbedding, */
          /* *boost::static_pointer_cast<mblas::Matrix>(SourceContext_)); */

      /* for (size_t j = 0; j < w.size(); ++j) { */
        /* if (batch[lengthIndex][j]) { */
          /* float p = Probs(j, w[j]); */
          /* scores[j] += log(p); */
        /* } */
      /* } */

      /* decoder_->Lookup(Embedding, w); */

      /* mblas::Swap(State, PrevState); */
      /* mblas::Swap(Embedding, PrevEmbedding); */
      /* ++lengthIndex; */
    /* } */

    /* for (int i = 0; i < scores.size(); ++i) { */
      /* nBestScores.push_back(scores[i]); */
    /* } */
  /* } */
  /* return nBestScores; */
/* } */


