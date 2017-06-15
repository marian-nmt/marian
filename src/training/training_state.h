#pragma once

#include <vector>

#include "common/definitions.h"
#include "training/config.h"

namespace marian {

class TrainingState;

class TrainingObserver {
public:
  virtual void epochHasChanged(TrainingState& state) = 0;
};

class TrainingState {
public:
  int epoch;
  int maxStalled;
  float eta;

  TrainingState(Ptr<Config> options)
      : epoch(1), maxStalled(0), eta(options->get<float>("learn-rate")) {}

  void registerObserver(Ptr<TrainingObserver> observer) {
    observers_.push_back(observer);
  }

  int next() {
    ++epoch;
    for (auto observer : observers_)
      observer->epochHasChanged(*this);
    return epoch;
  }

private:
  std::vector<Ptr<TrainingObserver>> observers_;

};
}
