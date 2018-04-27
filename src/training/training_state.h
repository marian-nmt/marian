#pragma once

#include <fstream>
#include <vector>

#include "common/definitions.h"

namespace marian {

class TrainingState;

class TrainingObserver {
public:
  virtual void init(TrainingState& state) {}
  virtual void actAfterEpoch(TrainingState& state) {}
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
  // The number of batches seen in this epoch
  size_t batchesEpoch{0};
  // The number of samples seen in this epoch
  size_t samples{0};

  // The number of stalled validations
  size_t stalled{0};
  // The largest number of stalled validations so far
  size_t maxStalled{0};
  // Last best validation score
  float validBest{0.f};
  std::string validator;
  // List of validators
  YAML::Node validators;
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

  // The state of the random number generator from a batch generator
  std::string seedBatch;
  // The state of the random number generator from a corpus
  std::string seedCorpus;

  bool loaded{false};
  bool validated{false};

  TrainingState(float learnRate) : eta(learnRate) {}

  void registerObserver(Ptr<TrainingObserver> observer) {
    observers_.push_back(observer);
    observers_.back()->init(*this);
  }

  void newEpoch() {
    ++epochs;
    for(auto observer : observers_)
      observer->actAfterEpoch(*this);
    samples = 0;
    batchesEpoch = 0;
  }

  void newBatch() {
    ++batches;
    ++batchesEpoch;
    loaded = false;
    validated = false;
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
    loaded = true;
    for(auto observer : observers_)
      observer->actAfterLoaded(*this);
  }

  void load(const std::string& name) {
    if(!boost::filesystem::exists(name))
      return;

    YAML::Node config = YAML::LoadFile(name);

    epochs = config["epochs"].as<size_t>();
    batches = config["batches"].as<size_t>();
    batchesEpoch = config["batches-epoch"].as<size_t>();
    samples = config["samples"].as<size_t>();

    stalled = config["stalled"].as<size_t>();
    maxStalled = config["stalled-max"].as<size_t>();
    validBest = config["valid-best"].as<float>();
    validator = config["validator"].as<std::string>();
    validators = config["validators"];
    reset = config["reset"].as<bool>();

    eta = config["eta"].as<float>();
    factor = config["eta-factor"].as<float>();
    warmupStart = config["warmup-start"].as<size_t>();

    costSum = config["cost-sum"].as<float>();
    samplesDisp = config["disp-samples"].as<size_t>();
    wordsDisp = config["disp-words"].as<size_t>();

    seedBatch = config["seed-batch"].as<std::string>();
    seedCorpus = config["seed-corpus"].as<std::string>();
  }

  void save(const std::string& name) {
    std::ofstream fout(name);
    YAML::Node config;

    config["epochs"] = epochs;
    config["batches"] = batches;
    config["batches-epoch"] = batchesEpoch;
    config["samples"] = samples;

    config["stalled"] = stalled;
    config["stalled-max"] = maxStalled;
    config["valid-best"] = validBest;
    config["validator"] = validator;
    config["validators"] = validators;
    config["reset"] = reset;

    config["eta"] = eta;
    config["eta-factor"] = factor;
    config["warmup-start"] = warmupStart;

    config["cost-sum"] = costSum;
    config["disp-samples"] = samplesDisp;
    config["disp-words"] = wordsDisp;

    config["seed-batch"] = seedBatch;
    config["seed-corpus"] = seedCorpus;

    fout << config;
  }

private:
  std::vector<Ptr<TrainingObserver>> observers_;
};
}
