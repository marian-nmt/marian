#pragma once

#include <vector>

#include "common/definitions.h"

namespace marian {

class TrainingState;

class TrainingObserver {
public:
  virtual void actAfterEpoch(TrainingState& state) {};
  virtual void actAfterBatches(TrainingState& state) {}
  virtual void actAfterStalled(TrainingState& state) {}
  virtual void actAfterLoaded(TrainingState& state) {}
};

class TrainingState {
public:
  // Current epoch
  size_t epochs{1};
  // The total number of batches
  size_t batches{0};
  // The number of samples seen in this epoch
  size_t samples{0};

  // The number of stalled validations
  size_t stalled{0};
  // The largest number of stalled validations so far
  size_t maxStalled{0};
  // Last best validation score
  float validBest{0.f};
  // Reset optimizer parameters
  bool reset{false};

  // Learning rate
  float eta;
  // Multiplication factor for learning rate
  float factor{1.f};
  size_t warmupStart{0};

  // Sum of costs since last display
  float costSum{0};
  // Number of samples seen since last display
  size_t samplesDisp{0};
  // Number of words seen since last display
  size_t wordsDisp{0};

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

  void newLoad() {
    for(auto observer : observers_)
      observer->actAfterLoaded(*this);
  }

  void load(const YAML::Node& config) {
    epochs = config["progress"]["epochs"].as<size_t>();
    batches = config["progress"]["batches"].as<size_t>();
    samples = config["progress"]["samples"].as<size_t>();

    stalled = config["progress"]["stalled"].as<size_t>();
    maxStalled = config["progress"]["stalled-max"].as<size_t>();
    validBest = config["progress"]["valid-best"].as<float>();
    reset = config["progress"]["reset"].as<bool>();

    eta = config["progress"]["eta"].as<float>();
    factor = config["progress"]["eta-factor"].as<float>();
    warmupStart = config["progress"]["warmup-start"].as<size_t>();

    costSum = config["progress"]["cost-sum"].as<float>();
    samplesDisp = config["progress"]["disp-samples"].as<size_t>();
    wordsDisp = config["progress"]["disp-words"].as<size_t>();
  }

  void save(YAML::Node& config) {
    config["progress"]["epochs"] = epochs;
    config["progress"]["batches"] = batches;
    config["progress"]["samples"] = samples;

    config["progress"]["stalled"] = stalled;
    config["progress"]["stalled-max"] = maxStalled;
    config["progress"]["valid-best"] = validBest;
    config["progress"]["reset"] = reset;

    config["progress"]["eta"] = eta;
    config["progress"]["eta-factor"] = factor;
    config["progress"]["warmup-start"] = warmupStart;

    config["progress"]["cost-sum"] = costSum;
    config["progress"]["disp-samples"] = samplesDisp;
    config["progress"]["disp-words"] = wordsDisp;
  }

private:
  std::vector<Ptr<TrainingObserver>> observers_;
};
}
