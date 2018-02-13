#pragma once

#include <vector>

#include "common/definitions.h"

namespace marian {

class TrainingState;

class TrainingObserver {
public:
  virtual void actAfterEpoch(TrainingState& state) = 0;
  virtual void actAfterBatches(TrainingState& state) {}
  virtual void actAfterStalled(TrainingState& state) {}
};

class TrainingState {
public:
  size_t epochs{1};
  size_t batches{0};
  size_t samples{0};

  size_t stalled{0};
  size_t maxStalled{0};
  size_t warmupStart{0};

  float eta;
  float factor{1.f};

  float costSum{0};
  size_t samplesDisp{0};
  size_t wordsDisp{0};
  bool reset{false};

  TrainingState(float learnRate) : eta(learnRate) {}

  void registerObserver(Ptr<TrainingObserver> observer) {
    observers_.push_back(observer);
  }

  void newEpoch() {
    ++epochs;
    for(auto observer : observers_)
      observer->actAfterEpoch(*this);
  }

  void newBatch() {
    ++batches;
    for(auto observer : observers_)
      observer->actAfterBatches(*this);
  }

  void newStalled(int num) {
    stalled = num;
    if(num > maxStalled)
      ++maxStalled;
    for(auto observer : observers_)
      observer->actAfterStalled(*this);
  }

  void load(const YAML::Node& config) {
    epochs = config["progress"]["epochs"].as<size_t>();
    batches = config["progress"]["batches"].as<size_t>();
    samples = config["progress"]["samples"].as<size_t>();

    stalled = config["progress"]["stalled"].as<size_t>();
    maxStalled = config["progress"]["stalled-max"].as<size_t>();
    warmupStart = config["progress"]["warmup-start"].as<size_t>();

    eta = config["progress"]["eta"].as<float>();
    factor = config["progress"]["eta-factor"].as<float>();

    costSum = config["progress"]["cost-sum"].as<float>();
    samplesDisp = config["progress"]["disp-samples"].as<size_t>();
    wordsDisp = config["progress"]["disp-words"].as<size_t>();
    reset = config["progress"]["reset"].as<bool>();
  }

  void save(YAML::Node& config) {
    config["progress"]["epochs"] = epochs;
    config["progress"]["batches"] = batches;
    config["progress"]["samples"] = samples;

    config["progress"]["stalled"] = stalled;
    config["progress"]["stalled-max"] = maxStalled;
    config["progress"]["warmup-start"] = warmupStart;

    config["progress"]["eta"] = eta;
    config["progress"]["eta-factor"] = factor;

    config["progress"]["cost-sum"] = costSum;
    config["progress"]["disp-samples"] = samplesDisp;
    config["progress"]["disp-words"] = wordsDisp;
    config["progress"]["reset"] = reset;
  }

private:
  std::vector<Ptr<TrainingObserver>> observers_;
};
}
