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
#include "common/sentence.h"

using namespace GPU;

void MosesPlugin::initGod(const std::string& configPath) {
  std::string configs = "-c " + configPath;
  god_.Init(configs);

  DeviceInfo deviceInfo = god_.GetNextDevice();  
  scorers_ = god_.GetScorers(deviceInfo);
  bestHyps_ = god_.GetBestHyps(deviceInfo);
}

MosesPlugin::MosesPlugin()
  : debug_(false),
    states_(new States()),
    firstWord_(true)
{}

MosesPlugin::~MosesPlugin()
{
}

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


void MosesPlugin::GeneratePhrases(const States& states, size_t lastWord, size_t numPhrases,
                                  std::vector<NeuralPhrase>& phrases) {
  assert(states.size() == scorers_.size());
  Histories histories(god_, sentences_);

  size_t batchSize = 1;
  std::vector<size_t> beamSizes(batchSize, 1);

  Beam prevHyps(batchSize, HypothesisPtr(new Hypothesis()));
  for (size_t i = 0; i < histories.size(); ++i) {
    histories.at(i)->Add(prevHyps);
  }

  States nextStates(scorers_.size());
  for (size_t i = 0; i < scorers_.size(); ++i){
    nextStates[i].reset(scorers_[i]->NewState());
  }
  size_t vocabSize = scorers_[0]->GetVocabSize();

  size_t maxLength = 5;

  for (size_t decoderStep = 0; decoderStep < maxLength; ++decoderStep) {
    for (size_t i = 0; i < scorers_.size(); i++) {
      Scorer &scorer = *scorers_[i];
      State &state = *states[i];
      State &nextState = *nextStates[i];

      scorer.Score(god_, state, nextState, beamSizes);
    }

    if (decoderStep == 0) {
      for (auto& beamSize : beamSizes) {
        beamSize = god_.Get<size_t>("beam-size");
      }
    }

    Beams beams(batchSize);

    (*bestHyps_)(god_, beams, prevHyps, beamSizes, scorers_, filterIndices_, true);

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

  const NBestList &nbl = histories.at(0)->NBest(god_.Get<size_t>("beam-size"));

  for (size_t i = 0; i < nbl.size(); ++i) {
    const Result& result = nbl[i];
    auto words = god_.Postprocess(god_.GetTargetVocab()(result.first));
    auto& scores = result.second->GetCostBreakdown();

    phrases.emplace_back(result.first, scores, 0, 1);
  }

}

States MosesPlugin::GenerateStates(const States& ParentStates,
                                   size_t lastWord,
                                   std::vector<size_t>& phrase) {
  Histories histories(god_, sentences_);

  size_t batchSize = 1;
  std::vector<size_t> beamSizes(batchSize, 1);

  Beam prevHyps(batchSize, HypothesisPtr(new Hypothesis()));
  for (size_t i = 0; i < histories.size(); ++i) {
    histories.at(i)->Add(prevHyps);
  }


  States states(scorers_.size());
  States nextStates(scorers_.size());
  for (size_t i = 0; i < scorers_.size(); ++i) {
    nextStates[i].reset(scorers_[i]->NewState());
    states[i].reset(scorers_[i]->NewState());
  }

  Beam survivors;
  for (size_t i = 0; i < batchSize; ++i) {
    survivors.emplace_back(new Hypothesis(prevHyps[i], lastWord, 0, 0.f));
  }

  for (size_t i = 0; i < scorers_.size(); i++) {
    scorers_[i]->AssembleBeamState(*ParentStates[i], survivors, *states[i]);
  }

  prevHyps.swap(survivors);

  size_t vocabSize = scorers_[0]->GetVocabSize();

  for (size_t decoderStep = 0; decoderStep < phrase.size(); ++decoderStep) {
    for (size_t i = 0; i < scorers_.size(); i++) {
      Scorer &scorer = *scorers_[i];
      State &state = *states[i];
      State &nextState = *nextStates[i];

      scorer.Score(god_, state, nextState, beamSizes);
    }

    Beam survivors;
    survivors.emplace_back(new Hypothesis(prevHyps[0], phrase[decoderStep], 0, 0.0f));

    for (size_t i = 0; i < scorers_.size(); i++) {
      scorers_[i]->AssembleBeamState(*nextStates[i], survivors, *states[i]);
    }

    prevHyps.swap(survivors);
  }

  for (auto scorer : scorers_) {
	  scorer->CleanUpAfterSentence();
  }

  return states;

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

size_t MosesPlugin::TargetVocab(const std::string& str) {
  return god_.GetTargetVocab()[str];
}

size_t MosesPlugin::SourceVocab(const std::string& str) {
  return god_.GetSourceVocab(0)[str];
}

States MosesPlugin::SetSource(const std::vector<size_t>& words) {
  if (sentences_.size() == 0) {
      sentences_.push_back(SentencePtr(new Sentence(god_, 0, words)));
  } else {
      sentences_.at(0).reset(new Sentence(god_, 0, words));
  }

  States states(scorers_.size());

  for (size_t i = 0; i < scorers_.size(); ++i) {
    states[i].reset(scorers_[i]->NewState());
    scorers_[i]->SetSource(sentences_);
    scorers_[i]->BeginSentenceState(*states[i], sentences_.size());
  }

  return states;
}

void MosesPlugin::Rescore(std::vector<HypoInfo> &hypos)
{

  for (HypoInfo &hypo : hypos) {
    States nextStates = GenerateStates(hypo.prevStates, hypo.lastWord, hypo.words);
    hypo.nextStates = nextStates;
  }
}
