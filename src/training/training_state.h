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
  int epochs{1};
  int batches{0};
  int stalled{0};
  int maxStalled{0};
  int warmupStart{0};
  float eta;
  float factor{1.f};
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
    stalled = config["progress"]["stalled"].as<size_t>();
    maxStalled = config["progress"]["stalled-max"].as<size_t>();
    warmupStart = config["progress"]["warmup-start"].as<size_t>();
    eta = config["progress"]["eta"].as<float>();
    factor = config["progress"]["eta-factor"].as<float>();
    reset = config["progress"]["reset"].as<bool>();
  }

  void save(YAML::Node& config) {
    config["progress"]["epochs"] = epochs;
    config["progress"]["batches"] = batches;
    config["progress"]["stalled"] = stalled;
    config["progress"]["stalled-max"] = maxStalled;
    config["progress"]["warmup-start"] = warmupStart;
    config["progress"]["eta"] = eta;
    config["progress"]["eta-factor"] = factor;
    config["progress"]["reset"] = reset;
  }

private:
  std::vector<Ptr<TrainingObserver>> observers_;
};
}
