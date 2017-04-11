#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <cmath>

#include "nmt.h"
#include "common/sentences.h"
#include "common/vocab.h"
#include "common/scorer.h"
#include "common/types.h"
#include "common/god.h"
#include <cuda.h>
#include <thrust/device_vector.h>

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

NMT::NMT()
: debug_(false),
  states_(new States()),
  firstWord_(true)
{
  auto deviceInfo_ = god_->GetNextDevice();
  scorers_ = god_->GetScorers(deviceInfo_);
}


NMT::NMT(std::vector<ScorerPtr>& scorers)
  : debug_(false),
    scorers_(scorers),
    states_(new States()),
    firstWord_(true)
{
}

NMT::~NMT() {
  SetDevice();
}

void NMT::Clean() {
  god_->Cleanup();
}


void NMT::ClearStates()
{
  firstWord_ = true;
  states_->clear();
}

void NMT::SetDevice() {
  const DeviceInfo& deviceInfo = scorers_[0]->GetDeviceInfo();
  if (deviceInfo.deviceType == GPUDevice) {
    cudaSetDevice(deviceInfo.deviceId);
  }
}


States NMT::CalcSourceContext(const std::vector<std::string>& srcWords)
{
  Sentences sentences;
  sentences.push_back(SentencePtr(new Sentence(*god_, 0, srcWords)));

  States states = NewStates();

  for (size_t i = 0; i < scorers_.size(); ++i) {
    scorers_[i]->SetSource(sentences);
    scorers_[i]->BeginSentenceState(*states[i], sentences.size());
  }

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


std::vector<ScorerPtr> NMT::NewScorers()
{
  auto deviceInfo = god_->GetNextDevice();
  auto scorers = god_->GetScorers(deviceInfo);
  return scorers;
}


size_t NMT::TargetVocab(const std::string& str)
{
  return god_->GetTargetVocab()[str];
}


void NMT::OnePhrase(
  const std::vector<std::string>& phrase,
  const States& inputStates,
  float& prob,
  size_t& unks,
  States& outputStates)
{
  States prevStates = NewStates();
  States nextStates = NewStates();


  for (size_t wordIdx = 0; wordIdx < phrase.size(); ++wordIdx) {
    size_t id = god_->GetTargetVocab()[phrase[wordIdx]];
    if(id == 1) {
      unks++;
    }

    for (size_t i = 0; i < scorers_.size(); i++) {
      Scorer &scorer = *scorers_[i];
      State &state = (wordIdx == 0) ? *inputStates[i] : *prevStates[i];
      State &nextState = *nextStates[i];

      scorer.Decode(*god_, state, nextState, {1});
      prob += scorer.GetProbs().GetValue(0, id);
      Beam survivor;
      survivor.emplace_back(new Hypothesis(nullptr, id, 0, prob));
      scorers_[i]->AssembleBeamState(*nextStates[i], survivor, state);
    }
  }
  std::swap(nextStates, outputStates);
}

void NMT::BatchSteps(const Batches& batches,
                     Scores& probsOut,
                     Scores& unksOut,
                     std::vector<States>& inputStates)
{

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

      scorer.Decode(*god_, state, nextState, {previousIds.size()});
    }

    std::vector<std::pair<int, int>> indices;
    for (size_t i = 0; i < previousIds.size(); ++i) {
      indices.push_back(std::make_pair(i, batches[batchIdx][previousIds[i]]));
    }

    for (auto& scorer : scorers_) {
      auto logProbs = scorer->GetProbs().GetScores(indices);
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


std::vector<double> NMT::RescoreNBestList(
    const std::vector<std::string>& nbest,
    const size_t maxBatchSize)
{
  std::vector<double> nBestScores;

  NBest nBest(nbest, god_->GetTargetVocab(), maxBatchSize);
  for (auto& batch: nBest.SplitNBestListIntoBatches()) {
    States prevStates = NewStates();
    States nextStates = NewStates();

    std::vector<size_t> previousIds;
    for (size_t i = 0; i < batch[0].size(); ++i) {
      previousIds.push_back(i);
    }

    for (size_t i = 0; i < scorers_.size(); ++i) {
      scorers_[i]->BeginSentenceState(*prevStates[i], {1});
    }

    size_t batchSize = batch[0].size();

    std::vector<float> scores(batch[0].size(), 0.0f);

    for (size_t batchIdx = 0; batchIdx < batch.size(); ++batchIdx) {
      for (size_t i = 0; i < scorers_.size(); i++) {
        Scorer &scorer = *scorers_[i];
        const State &state =  *prevStates[i];
        State &nextState = *nextStates[i];

        if (batchIdx == 0) {
          scorer.Decode(*god_, state, nextState, {1});
        } else {
          scorer.Decode(*god_, state, nextState, {previousIds.size()});
        }
      }

      std::vector<std::pair<int, int>> indices;
      for (size_t i = 0; i < previousIds.size(); ++i) {
        if (batchIdx == 0) {
          indices.push_back(std::make_pair(0, batch[batchIdx][previousIds[i]]));
        } else {
          indices.push_back(std::make_pair(i, batch[batchIdx][previousIds[i]]));
        }
      }

      for (auto& scorer : scorers_) {
        auto logProbs = scorer->GetProbs().GetScores(indices);
        for (size_t i = 0; i < previousIds.size(); ++i) {
            scores[previousIds[i]] += logProbs[i];
        }
      }

      for (size_t ii = 0; ii < scores.size(); ++ii) std::cerr << scores[ii] << " ";
      std::cerr << std::endl;

      std::vector<size_t> nextIds;
      std::vector<size_t> nextHypIds;
      for (size_t i = 0; i < previousIds.size(); ++i) {
        if (batch[batchIdx + 1][previousIds[i]] != -1) {
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
        if (batchIdx == 0) {
          survivors.emplace_back(new Hypothesis(nullptr, batch[batchIdx][previousIds[i]], 0, 0.0f));
        } else {
          survivors.emplace_back(new Hypothesis(nullptr, batch[batchIdx][previousIds[i]], nextHypIds[i], 0.0f));
        }
      }

      for (size_t i = 0; i < scorers_.size(); ++i) {
        scorers_[i]->AssembleBeamState(*nextStates[i], survivors, *prevStates[i]);
      }
    }

    for (auto& score : scores) {
      nBestScores.push_back(score);
    }
  }
  return nBestScores;
}
