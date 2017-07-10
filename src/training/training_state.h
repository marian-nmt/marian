#pragma once

#include <vector>

#include "common/config.h"
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
  float eta;

  TrainingState(Ptr<Config> options) : eta(options->get<float>("learn-rate")) {}

  void registerObserver(Ptr<TrainingObserver> observer) {
    observers_.push_back(observer);
  }

  void newEpoch(int num) {
    epochs = num;
    for(auto observer : observers_)
      observer->actAfterEpoch(*this);
  }

  void newBatches(int num) {
    batches = num;
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

private:
  std::vector<Ptr<TrainingObserver>> observers_;
};
}
