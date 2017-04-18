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
#include "common/history.h"
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

size_t NMT::GetBatchSize() {
  return god_->Get<size_t>("beam-size");
}

NMT::NMT()
: debug_(false),
  firstWord_(true)
{
  auto deviceInfo_ = god_->GetNextDevice();
  std::cerr << "Device ID: " << deviceInfo_.deviceId << std::endl;
  if (deviceInfo_.deviceType == GPUDevice) {
    cudaSetDevice(deviceInfo_.deviceId);
  }
  scorers_ = god_->GetScorers(deviceInfo_);
  bestHyps_ = god_->GetBestHyps(deviceInfo_);
}


NMT::NMT(std::vector<ScorerPtr>& scorers)
  : debug_(false),
    firstWord_(true)
{
  const DeviceInfo& deviceInfo = scorers[0]->GetDeviceInfo();
  std::cerr << "Device ID: " << deviceInfo.deviceId << std::endl;
  if (deviceInfo.deviceType == GPUDevice) {
    cudaSetDevice(deviceInfo.deviceId);
  }
  scorers_ = scorers;
}

NMT::~NMT() {
  std::cerr << "Cleaning NMT~~" << std::endl;
  SetDevice();
}

void NMT::Clean() {
  std::cerr << "Cleaning GOD from NMT" << std::endl;
  god_->Cleanup();
}


void NMT::ClearStates()
{
  firstWord_ = true;
  // states_->clear();
}

void NMT::SetDevice() {
  const DeviceInfo& deviceInfo = scorers_[0]->GetDeviceInfo();
  if (deviceInfo.deviceType == GPUDevice) {
    cudaSetDevice(deviceInfo.deviceId);
  }
}


States NMT::CalcSourceContext(const std::vector<std::string>& srcWords)
{
  // std::cerr << "Setting SRC " << std::endl;
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


void NMT::BatchSteps(const Batches& batches,
                     Scores& probsOut,
                     Scores& unksOut,
                     std::vector<States>& inputStates)
{
  SetDevice();
  // std::cerr << "BatchStesp " << std::endl;
  States prevStates = NewStates();
  States nextStates = NewStates();

  std::vector<States> tmp(scorers_.size());
  for (auto& states : inputStates) {
    for (size_t scorerIdx = 0; scorerIdx < scorers_.size(); ++scorerIdx) {
      tmp[scorerIdx].push_back(states[scorerIdx]);
    }
  }

  // std::cerr << "joining steps " << std::endl;
  for (size_t scorerIdx = 0; scorerIdx < scorers_.size(); ++scorerIdx) {
    prevStates[scorerIdx]->JoinStates(tmp[scorerIdx]);
  }

  // std::cerr _<< prevStates[0]->Debug() << std::endl;

  std::vector<size_t> previousIds;
  for (size_t i = 0; i < batches[0].size(); ++i) {
    previousIds.push_back(i);
  }


  for (size_t batchIdx = 0; batchIdx < batches.size(); ++batchIdx) {
    for (size_t i = 0; i < scorers_.size(); i++) {
      Scorer &scorer = *scorers_[i];
      const State &state =  *prevStates[i];
      State &nextState = *nextStates[i];

      // std::cerr << "Decoding " << std::endl;
      std::vector<size_t> beamSizes(1, previousIds.size());
      scorer.Decode(*god_, state, nextState, beamSizes);
      // std::cerr << "Decoding DONE" << std::endl;
    }

    std::vector<std::pair<int, int>> indices;
    for (size_t i = 0; i < previousIds.size(); ++i) {
      if (batches[batchIdx][previousIds[i]] >= scorers_[0]->GetVocabSize()) {
        indices.push_back(std::make_pair(i, 1));
      } else {
        indices.push_back(std::make_pair(i, batches[batchIdx][previousIds[i]]));
      }
      // std::cerr << "(" << i << " : " << batches[batchIdx][previousIds[i]] << ") " ;
    }
    // std::cerr << std::endl;

    for (auto& scorer : scorers_) {
    // std::cerr << "GETTING SCORES: " <<scorer->GetProbs().Rows() << " " << scorer->GetProbs().Cols() << std::endl;
      auto logProbs = scorer->GetProbs().GetScores(indices);
      // std::cerr << "Got log scores" << std::endl;
      for (size_t i = 0; i < previousIds.size(); ++i) {
          probsOut[previousIds[i]] += logProbs[i];
      }
    }
    // std::cerr << "GETTING SCORES: DONE" << std::endl;

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
  // std::cerr << "RESCORING" << std::endl;
  std::vector<double> nBestScores;

  NBest nBest(nbest, god_->GetTargetVocab(), GetBatchSize());
  for (auto& batch: nBest.SplitNBestListIntoBatches()) {
    // std::cerr << "REscoging batch" << std::endl;
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
      // std::cerr << "REscoging " << batchIdx << "/" << batch.size() << std::endl;
      for (size_t i = 0; i < scorers_.size(); i++) {
        Scorer &scorer = *scorers_[i];
        const State &state =  *prevStates[i];
        State &nextState = *nextStates[i];

        // std::cerr << "Decoding ... " << previousIds.size() << std::endl;
        // for (auto v : previousIds) std::cerr <<v << " ";
        // std::cerr << std::endl;
        if (batchIdx == 0) {
          scorer.Decode(*god_, state, nextState, {1});
        } else {
          scorer.Decode(*god_, state, nextState, {previousIds.size()});
        }
      }

      // std::cerr << "Filling indices" << std::endl;
      std::vector<std::pair<int, int>> indices;
      for (size_t i = 0; i < previousIds.size(); ++i) {
        if (batchIdx == 0) {
          indices.push_back(std::make_pair(0, batch[batchIdx][previousIds[i]]));
        } else {
          indices.push_back(std::make_pair(i, batch[batchIdx][previousIds[i]]));
        }
      }

      // std::cerr << "Getting scores" << std::endl;
      for (auto& scorer : scorers_) {
        auto logProbs = scorer->GetProbs().GetScores(indices);
        for (size_t i = 0; i < previousIds.size(); ++i) {
            scores[previousIds[i]] += logProbs[i];
        }
      }

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

      // std::cerr << "Assembling..." << std::endl;
      for (size_t i = 0; i < scorers_.size(); ++i) {
        scorers_[i]->AssembleBeamState(*nextStates[i], survivors, *prevStates[i]);
      }
    }

    for (auto& score : scores) {
      nBestScores.push_back(score);
    }
  }
  // std::cerr << "RESCORING DONE" << std::endl;
  return nBestScores;
}

std::vector<NeuralExtention> NMT::GetNeuralExtentions(std::vector<States>& inputStates) {
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

      scorer.Decode(*god_, state, nextState, {inputStates.size()});
    }

    bool returnAlignment = true;

    Beams beams;
    std::vector<size_t> beamSizes = {batchSize};
    bestHyps_->CalcBeam(*god_, prevHyps, scorers_, filterIndices_,
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
            if (alignment->at(i) >= 0.3f) {
                align.push_back(i);
            }
        }
        output.emplace_back(phrase, cost, align);
      }
    }
  }

  return output;
}
