#include "plugin/nmt.h"

#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <string>
#include <vector>
#include <cmath>
#include <cuda.h>

#include "common/sentences.h"
#include "common/vocab.h"
#include "common/scorer.h"
#include "common/types.h"
#include "common/god.h"
#include "common/history.h"

using namespace amunmt;

std::shared_ptr<God> NMT::god_ = nullptr;

void NMT::InitGod(const std::string& configFilePath) {
  std::string argv = "-c " + configFilePath;
  god_.reset(new God());
  god_->Init(argv);
}

size_t NMT::GetTotalThreads()
{
  return god_->GetTotalThreads();
}

size_t NMT::GetBatchSize() {
  return god_->Get<size_t>("beam-size");
}

NMT::NMT()
{
  auto deviceInfo_ = god_->GetNextDevice();
  // std::cerr << "Device ID: " << deviceInfo_.deviceId << std::endl;
  if (deviceInfo_.deviceType == GPUDevice) {
    cudaSetDevice(deviceInfo_.deviceId);
  }
  scorers_ = god_->GetScorers(deviceInfo_);
  bestHyps_ = god_->GetBestHyps(deviceInfo_);
}


NMT::~NMT() {
  SetDevice();
}

void NMT::Clean() {
  god_->Cleanup();
}

void NMT::SetDevice() {
  const DeviceInfo& deviceInfo = scorers_[0]->GetDeviceInfo();
  if (deviceInfo.deviceType == GPUDevice) {
    cudaSetDevice(deviceInfo.deviceId);
  }
}


States NMT::CalcSourceContext(const std::vector<std::string>& srcWords)
{
  // std::cerr << "Setting source sentence..." << std::endl;
  Sentences sentences;
  sentences.push_back(SentencePtr(new Sentence(*god_, 0, srcWords)));

  States states = NewStates();

  for (size_t i = 0; i < scorers_.size(); ++i) {
    scorers_[i]->SetSource(sentences);
    scorers_[i]->BeginSentenceState(*states[i], sentences.size());
    // std::cerr << "SRC: " << states[i]->Debug() << std::endl;
  }

  // std::cerr << "Setting source sentence... DONE" << std::endl;
  return states;
}


States NMT::NewStates() const
{
  size_t numScorers = scorers_.size();

  States states(numScorers);
  for (size_t i = 0; i < numScorers; i++) {
    Scorer &scorer = *scorers_[i];
    states[i].reset(scorer.NewState());
  }

  return states;
}


size_t NMT::TargetVocab(const std::string& str)
{
  return god_->GetTargetVocab()[str];
}


void NMT::BatchSteps(const Batches& batches,
                     Scores& probsOut,
                     std::vector<States>& inputStates)
{
  SetDevice();
  States prevStates = NewStates();
  States nextStates = NewStates();

  std::vector<States> tmp(scorers_.size());
  for (auto& states : inputStates) {
    for (size_t scorerIdx = 0; scorerIdx < scorers_.size(); ++scorerIdx) {
      tmp[scorerIdx].push_back(states[scorerIdx]);
    }
  }

  for (size_t scorerIdx = 0; scorerIdx < scorers_.size(); ++scorerIdx) {
    prevStates[scorerIdx]->JoinStates(tmp[scorerIdx]);
  }

  std::vector<size_t> previousIds;
  for (size_t i = 0; i < batches[0].size(); ++i) {
    previousIds.push_back(i);
  }


  for (size_t batchIdx = 0; batchIdx < batches.size(); ++batchIdx) {
    for (size_t i = 0; i < scorers_.size(); i++) {
      Scorer &scorer = *scorers_[i];
      const State &state =  *prevStates[i];
      State &nextState = *nextStates[i];

      std::vector<size_t> beamSizes(1, previousIds.size());
      scorer.Decode(state, nextState, beamSizes);
    }

    std::vector<std::pair<size_t, size_t>> indices;
    for (size_t i = 0; i < previousIds.size(); ++i) {
      if (batches[batchIdx][previousIds[i]] >= (int)scorers_[0]->GetVocabSize()) {
        indices.push_back(std::make_pair(i, 1));
      } else {
        indices.push_back(std::make_pair(i, batches[batchIdx][previousIds[i]]));
      }
    }

    for (auto& scorer : scorers_) {
      auto logProbs = scorer->GetScores(indices);
      for (size_t i = 0; i < previousIds.size(); ++i) {
          probsOut[previousIds[i]] += logProbs[i];
      }
    }

    std::vector<size_t> nextIds;
    std::vector<size_t> nextHypIds;
    for (size_t i = 0; i < previousIds.size(); ++i) {
      if (batches[batchIdx + 1][previousIds[i]] == -1) {
        for (size_t scorerIdx = 0; scorerIdx < scorers_.size(); ++scorerIdx) {
          Beam tBeam;
          tBeam.emplace_back(new Hypothesis(nullptr, batches[batchIdx][previousIds[i]], i, 0.0f));
          inputStates[previousIds[i]][scorerIdx].reset(scorers_[scorerIdx]->NewState());
          scorers_[scorerIdx]->AssembleBeamState(*nextStates[scorerIdx], tBeam, *inputStates[previousIds[i]][scorerIdx]);
        }
      } else {
        nextIds.push_back(previousIds[i]);
        nextHypIds.push_back(i);
      }
    }

    if (nextIds.empty()) {
      break;
    }

    previousIds.swap(nextIds);

    Beam survivors;
    for (size_t i = 0; i < previousIds.size(); ++i) {
      survivors.emplace_back(new Hypothesis(nullptr, batches[batchIdx][previousIds[i]], nextHypIds[i], 0.0f));
    }

    for (size_t i = 0; i < scorers_.size(); ++i) {
      scorers_[i]->AssembleBeamState(*nextStates[i], survivors, *prevStates[i]);
    }
  }
}


std::vector<float> NMT::RescoreNBestList(const std::vector<std::string>& nbest)
{
  // std::cerr << "Rescoring N-Best list..." << std::endl;
  States states = NewStates();
  for (size_t i = 0; i < scorers_.size(); ++i) {
    scorers_[i]->BeginSentenceState(*states[i], {1});
  }

  std::vector<States> inputStates(nbest.size(), states);
  NBest nBest(nbest, inputStates, god_->GetTargetVocab(), GetBatchSize());
  return Rescore(nBest, false);
}

void NMT::RescorePhrases(const std::vector<std::vector<std::string>>& phrases, std::vector<States>& inputStates, Scores& probs)
{
  // std::cerr << "Rescoring Phrases..." << std::endl;
  NBest nBest(phrases, inputStates, god_->GetTargetVocab(), GetBatchSize());
  auto scores = Rescore(nBest, true);
  std::swap(probs, scores);
  // std::cerr << "Rescoring Phrases..." << std::endl;
}

States NMT::JoinStates(const std::vector<States*>& inStates)
{
  // std::cerr << "Join States..." << std::endl;
  States prevStates = NewStates();
  std::vector<States> tmp(scorers_.size());
  for (auto& states : inStates) {
    for (size_t scorerIdx = 0; scorerIdx < scorers_.size(); ++scorerIdx) {
      tmp[scorerIdx].push_back(states->at(scorerIdx));
    }
  }

  for (size_t scorerIdx = 0; scorerIdx < scorers_.size(); ++scorerIdx) {
    prevStates[scorerIdx]->JoinStates(tmp[scorerIdx]);
  }
  return prevStates;
}

Beam NMT::GetSurvivors(RescoreBatch& rescoreBatch, size_t step) {
  Beam survivors;
  std::vector<size_t> nextHyps;
  for (size_t i = 0; i < rescoreBatch.prevIds[step].size(); ++i) {
    if (rescoreBatch.data[step + 1][rescoreBatch.prevIds[step][i]] != -1) {
      nextHyps.push_back(i);
    }
  }

  for (size_t i = 0; i < nextHyps.size(); ++i) {
     survivors.emplace_back(new Hypothesis(nullptr,
                                           rescoreBatch.data[step][rescoreBatch.prevIds[step + 1][i]],
                                           nextHyps[i],
                                           0.0f));
  }

  return survivors;
}


void NMT::SaveFinalStates(const States& inStates, size_t step, RescoreBatch& rescoreBatch) {
  for (auto i : rescoreBatch.completed[step]) {
    for (size_t scorerIdx = 0; scorerIdx < scorers_.size(); ++scorerIdx) {
      size_t outStateId = rescoreBatch.prevIds[step][i];
      size_t lastWord = rescoreBatch.data[step][outStateId];
      auto& outState = rescoreBatch.states[outStateId]->at(scorerIdx);

      Beam tBeam;
      tBeam.emplace_back(new Hypothesis(nullptr, lastWord, i, 0.0f));

      outState.reset(scorers_[scorerIdx]->NewState());
      scorers_[scorerIdx]->AssembleBeamState(*inStates[scorerIdx], tBeam, *outState);
    }
  }
}


std::vector<float> NMT::Rescore(NBest& nBest, bool returnFinalStates) {
  std::vector<float> scores;
  for (auto& rescoreBatch: nBest.SplitNBestListIntoBatches()) {
    States prevStates = JoinStates(rescoreBatch.states);
    States nextStates = NewStates();
    std::vector<float> probs(rescoreBatch.data[0].size());

    for (size_t stepIdx = 0; stepIdx < rescoreBatch.length(); ++stepIdx) {
      for (size_t ii = 0; ii < scorers_.size(); ii++) {
        Scorer &scorer = *scorers_[ii];
        const State &state =  *prevStates[ii];
        State &nextState = *nextStates[ii];

        scorer.Decode(state, nextState);

        auto logProbs = scorer.GetScores(rescoreBatch.indices[stepIdx]);
        for (size_t i = 0; i < rescoreBatch.prevIds[stepIdx].size(); ++i) {
          probs[rescoreBatch.prevIds[stepIdx][i]] += logProbs[i];
        }
      }

      if (returnFinalStates) {
          SaveFinalStates(nextStates, stepIdx, rescoreBatch);
      }

      Beam survivors = GetSurvivors(rescoreBatch, stepIdx);

      if (survivors.empty()) {
        break;
      }

      for (size_t i = 0; i < scorers_.size(); ++i) {
        scorers_[i]->AssembleBeamState(*nextStates[i], survivors, *prevStates[i]);
      }
    }

    for (auto score : probs) {
        scores.push_back(score);
    }
  }
  return scores;
}



std::vector<NeuralExtention> NMT::GetNeuralExtentions(const std::vector<States>& inputStates) {
  std::vector<NeuralExtention> output;
  States prevStates = NewStates();
  States nextStates = NewStates();

  size_t batchSize = inputStates.size();

  Beam prevHyps(batchSize, HypothesisPtr(new Hypothesis()));

  std::vector<States> tmp(scorers_.size());
  for (auto& states : inputStates) {
    for (size_t scorerIdx = 0; scorerIdx < scorers_.size(); ++scorerIdx) {
      tmp[scorerIdx].push_back(states[scorerIdx]);
    }
  }

  for (size_t scorerIdx = 0; scorerIdx < scorers_.size(); ++scorerIdx) {
    prevStates[scorerIdx]->JoinStates(tmp[scorerIdx]);
  }

  const size_t maxExtensionLength = 1;
  for (size_t step = 0; step < maxExtensionLength; ++step) {
    for (size_t i = 0; i < scorers_.size(); i++) {
      Scorer &scorer = *scorers_[i];
      const State &state =  *prevStates[i];
      State &nextState = *nextStates[i];

      scorer.Decode(state, nextState);
    }

    bool returnAlignment = true;

    Beams beams;
    std::vector<size_t> beamSizes = {batchSize};
    std::vector<size_t> filterIndices;

    bestHyps_->CalcBeam(*god_, prevHyps, scorers_, filterIndices,
                        returnAlignment, beams, beamSizes);

    for (auto& beam: beams) {
      for (auto& hyp : beam) {
        float cost = hyp->GetCost();

        std::vector<size_t> phrase;
        auto iter = hyp;
        while (iter != nullptr) {
            phrase.push_back(iter->GetWord());
            iter = iter->GetPrevHyp();
        }

        std::vector<size_t> align;
        auto alignment = hyp->GetAlignment(0);
        for (size_t i = 0; i < alignment->size(); ++i) {
          if ((*alignment)[i] >= 0.3f) {
            align.push_back(i);
          }
        }
        output.emplace_back(phrase, cost, align, hyp->GetPrevStateIndex());
      }
    }
  }

  return output;
}
