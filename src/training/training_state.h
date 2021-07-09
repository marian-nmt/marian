#pragma once

#include "common/definitions.h"
#include "common/filesystem.h"
#include "common/scheduling_parameter.h"
#include "common/utils.h"

#include <fstream>
#include <vector>

namespace marian {

class TrainingState;

class TrainingObserver {
public:
  virtual ~TrainingObserver() {}

  virtual void init(TrainingState&) {}
  virtual void actAfterEpoch(TrainingState&) {}
  virtual void actAfterBatches(TrainingState&) {}
  virtual void actAfterStalled(TrainingState&) {}
  virtual void actAfterLoaded(TrainingState&) {}
};

class TrainingState {
public:
  // Current epoch
  size_t epochs{1};
  // The total number of updates since beginning of training   --@TODO: rename to 'updates'
  size_t batches{0};
  // The number of batches seen in this epoch  --note: not updates; an update can consist of multiple batches
  size_t batchesEpoch{0};
  // The number of sentences seen in this epoch  --@TODO: rename to 'sentencesEpoch'
  size_t samplesEpoch{0};
  // Number of word labels processed since beginning of training
  size_t labelsTotal{0};

  // Values before previous update() call
  size_t prevLabelsTotal{0}, prevBatches{0}, prevEpochs{0};

  // The number of stalled validations
  size_t stalled{0};
  // The largest number of stalled validations so far
  size_t maxStalled{0};
  std::string validator;
  // List of validators
  YAML::Node validators;
  // Reset optimizer parameters
  bool reset{false};

  // Current learning rate, representing all adjustment processes and factors
  float eta;
  void updateEta(float dynamicBaseLR) { // note: no other function may write to 'eta' (besides load())
    eta = dynamicBaseLR * factor;
  }
  // State-based multiplication factor for learning rate
  float factor{1.f};
  // @TODO: should also have warmup period here?
  SchedulingParameter warmupStart; // has same unit as lr-warmup

  // Sum of costs since last display
  float costSum{0};
  // Number of labels aggregated in
  // costSum since last display
  float costCount{0};

  // Number of words seen since last display
  size_t wordsDisp{0};
  // Number of samples/sentences seen since last display
  size_t samplesDisp{0};
  // Number of updates seen since last display
  size_t updatesDisp{0};

  // Running average of gradient norm
  float gradientNormAvg{0};
  // Running variance of gradient norm
  float gradientNormVar{0};

  // Running average of log gradient norm
  float logGradientNormAvg{0};
  // Running variance of log gradient norm
  float logGradientNormVar{0};

  // The state of the random number generator from a batch generator
  std::string seedBatch;
  // The state of the random number generator from a corpus
  std::string seedCorpus;

  // Set flag if training was resumed
  bool loaded{false};

  // Set flag if the model was validated in the current batch
  bool validated{false};

  TrainingState(float learnRate) {
    updateEta(learnRate);
  }

  void registerObserver(Ptr<TrainingObserver> observer) {
    observer->init(*this);
    wObservers_.push_back(observer);
  }

  // return the totals count that corresponds to the given unit (batches, labels, or epochs)
  size_t getProgressIn(SchedulingUnit u) const {
    switch(u) {
      case SchedulingUnit::trgLabels: return labelsTotal;
      case SchedulingUnit::updates  : return batches;
      case SchedulingUnit::epochs   : return epochs;
      default: ABORT("corrupt enum value");
    }
  }

  // update() first calls this
  // This is to make sure that enteredNewPeriodOf() can detect a transition intoa new period
  void rememberPreviousProgress() {
    prevLabelsTotal = labelsTotal;
    prevBatches     = batches;
    prevEpochs      = epochs;
  }

  size_t getPreviousProgressIn(SchedulingUnit u) const {
    switch(u) {
      case SchedulingUnit::trgLabels: return prevLabelsTotal;
      case SchedulingUnit::updates  : return prevBatches;
      case SchedulingUnit::epochs   : return prevEpochs;
      default: ABORT("corrupt enum value");
    }
  }

  // Tests whether we entered a new period, e.g. disp-freq, according to the
  // unit in which that parameter is given. There are a few edge cases:
  //  - this function will be called many times within the same epoch
  //  - labelsTotal does not increment by 1, so simple modulus does not work
  //
  // So instead of modulus==0, this function compares the previous progress/period
  // to the current, and triggers if they differ (i.e. the border between two
  // periods was crossed). This requires that rememberPreviousProgress() is called
  // between calls to this. We call it from update(). Unfortunately, newEpoch()
  // is called at the wrong place for this to work, so SchedulingUnit::epoch is forbidden
  // for periods.
  bool enteredNewPeriodOf(std::string schedulingParam) const {
    auto period = SchedulingParameter::parse(schedulingParam);
    // @TODO: adapt to logical epochs
    ABORT_IF(period.unit == SchedulingUnit::epochs,
             "Unit {} is not supported for frequency parameters",
             schedulingParam);
    auto previousProgress = getPreviousProgressIn(period.unit);
    auto progress = getProgressIn(period.unit);
    return period && progress / period.n != previousProgress / period.n;
  }

  void newEpoch() {
    ++epochs;
    for(auto wObserver : wObservers_) {
      auto observer = wObserver.lock();
      ABORT_IF(!observer, "Training observer object expired. Make sure all registered observers exist during scheduler life time");
      observer->actAfterEpoch(*this);
    }
    samplesEpoch = 0;
    batchesEpoch = 0;
  }

  void newUpdate(size_t batchesInUpdate) {
    ++batches;
    batchesEpoch += batchesInUpdate;
    loaded = false;
    validated = false;
    for(auto wObserver : wObservers_) {
      auto observer = wObserver.lock();
      ABORT_IF(!observer, "Training observer object expired. Make sure all registered observers exist during scheduler life time");
      observer->actAfterBatches(*this);
    }
  }

  void newStalled(size_t num) {
    stalled = num;
    if(num > maxStalled)
      ++maxStalled;
    for(auto wObserver : wObservers_) {
      auto observer = wObserver.lock();
      ABORT_IF(!observer, "Training observer object expired. Make sure all registered observers exist during scheduler life time");
      observer->actAfterStalled(*this);
    }
  }

  void newLoad() {
    loaded = true;
    for(auto wObserver : wObservers_) {
      auto observer = wObserver.lock();
      ABORT_IF(!observer, "Training observer object expired. Make sure all registered observers exist during scheduler life time");
      observer->actAfterLoaded(*this);
    }
  }

  void load(const std::string& name) {
    if(!filesystem::exists(name))
      return;

    YAML::Node config = YAML::LoadFile(name);

    epochs = config["epochs"].as<size_t>();
    batches = config["batches"].as<size_t>();
    batchesEpoch = config["batches-epoch"].as<size_t>();
    // different serialization name for backward compatibility
    samplesEpoch = config["samples"].as<size_t>();

    // clang-format off
    // optional for backward compatibility
    labelsTotal     = config["labels-total"]      ? config["labels-total"].as<size_t>()      : 0;
    prevLabelsTotal = config["prev-labels-total"] ? config["prev-labels-total"].as<size_t>() : 0;
    prevBatches     = config["prev-batches"]      ? config["prev-batches"].as<size_t>()      : 0;
    prevEpochs      = config["prev-epochs"]       ? config["prev-epochs"].as<size_t>()       : 0;
    // clang-format on

    stalled = config["stalled"].as<size_t>();
    maxStalled = config["stalled-max"].as<size_t>();
    validator = config["validator"].as<std::string>();
    validators = config["validators"];
    reset = config["reset"].as<bool>();

    eta = config["eta"].as<float>();
    factor = config["eta-factor"].as<float>();

    warmupStart = SchedulingParameter::parse(config["warmup-start"].as<std::string>());

    costSum = config["cost-sum"].as<float>();
    costCount = config["cost-count"].as<float>();

    wordsDisp = config["disp-words"].as<size_t>();
    samplesDisp = config["disp-samples"].as<size_t>();
    updatesDisp = config["disp-updates"].as<size_t>();

    gradientNormAvg = config["gradient-norm-avg"].as<float>();
    gradientNormVar = config["gradient-norm-var"].as<float>();

    logGradientNormAvg = config["log-gradient-norm-avg"].as<float>();
    logGradientNormVar = config["log-gradient-norm-var"].as<float>();

    seedBatch = config["seed-batch"].as<std::string>();
    seedCorpus = config["seed-corpus"].as<std::string>();
  }

  void save(const std::string& name) const {
    std::ofstream fout(name);
    YAML::Node config;

    config["epochs"] = epochs;
    config["batches"] = batches;
    config["batches-epoch"] = batchesEpoch;
    config["samples"] = samplesEpoch;
    config["labels-total"] = labelsTotal;
    config["prev-labels-total"] = prevLabelsTotal;
    config["prev-batches"] = prevBatches;
    config["prev-epochs"] = prevEpochs;

    config["stalled"] = stalled;
    config["stalled-max"] = maxStalled;
    config["validator"] = validator;
    config["validators"] = validators;
    config["reset"] = reset;

    config["eta"] = eta;
    config["eta-factor"] = factor;
    config["warmup-start"] = std::string(warmupStart);

    config["cost-sum"] = costSum;
    config["cost-count"] = costCount;

    config["disp-updates"] = updatesDisp;
    config["disp-samples"] = samplesDisp;
    config["disp-words"] = wordsDisp;

    config["gradient-norm-avg"] = gradientNormAvg;
    config["gradient-norm-var"] = gradientNormVar;

    config["log-gradient-norm-avg"] = logGradientNormAvg;
    config["log-gradient-norm-var"] = logGradientNormVar;

    config["seed-batch"] = seedBatch;
    config["seed-corpus"] = seedCorpus;

    fout << config;
  }

  std::string fillTemplate(const std::string& templ) const {
    // The formatting below uses fmtlib, which is included with spdlog
    // and is included via the logger.
    return fmt::format(templ.c_str(),
                       fmt::arg("E", epochs),
                       fmt::arg("U", batches),
                       fmt::arg("B", batchesEpoch),
                       fmt::arg("T", labelsTotal));
  }

private:
  // this needs to be a vector of weak pointers, otherwise
  // it is likely to cause circular dependencies.
  std::vector<Weak<TrainingObserver>> wObservers_;

};
}  // namespace marian
