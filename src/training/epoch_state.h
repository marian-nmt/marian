#pragma once

#include <vector>

#include "common/definitions.h"
#include "training/config.h"

namespace marian {

class EpochState;

class EpochStateObserver {
public:
  virtual void epochHasChanged(EpochState& state) = 0;
};

class EpochState {
public:
  int epoch;
  float eta;

  EpochState(Ptr<Config> options)
      : epoch(1), eta(options->get<float>("learn-rate")) {}

  void registerObserver(Ptr<EpochStateObserver> observer) {
    observers_.push_back(observer);
  }

  int next() {
    ++epoch;
    for (auto observer : observers_)
      observer->epochHasChanged(*this);
    return epoch;
  }

private:
  std::vector<Ptr<EpochStateObserver>> observers_;

};
}
