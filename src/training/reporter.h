#pragma once

#include "training/config.h"
#include "training/training_state.h"
#include "training/validator.h"

namespace marian {

template <class DataSet>
class Reporter : public TrainingObserver {
private:
  YAML::Node progress;

  Ptr<Config> options_;
  std::vector<Ptr<Validator<DataSet>>> validators_;

  float costSum{0};

  size_t epochs{1};
  size_t samples{0};
  size_t samplesDisp{0};
  size_t wordsDisp{0};
  size_t batches{0};

  Ptr<TrainingState> trainState_;

  boost::timer::cpu_timer timer;

public:
  Reporter(Ptr<Config> options, Ptr<TrainingState> state)
      : options_(options), trainState_(state) {}

  bool keepGoing() {
    // stop if it reached the maximum number of epochs
    int stopAfterEpochs = options_->get<size_t>("after-epochs");
    if(stopAfterEpochs > 0 && epochs > stopAfterEpochs)
      return false;

    // stop if it reached the maximum number of batch updates
    int stopAfterBatches = options_->get<size_t>("after-batches");
    if(stopAfterBatches > 0 && batches >= stopAfterBatches)
      return false;

    // stop if the first validator did not improve for a given number of checks
    int stopAfterStalled = options_->get<size_t>("early-stopping");
    if(stopAfterStalled > 0 && !validators_.empty()
       && stalled() >= stopAfterStalled)
      return false;

    return true;
  }

  void increaseEpoch() {
    LOG(info, "Seen {} samples", samples);

    epochs++;
    trainState_->newEpoch();
    samples = 0;

    LOG(info, "Starting epoch {}", epochs);
  }

  void finished() { LOG(info, "Training finshed"); }

  void addValidator(Ptr<Validator<DataSet>> validator) {
    validators_.push_back(validator);
  }

  bool validating() {
    return (!validators_.empty()
            && batches % options_->get<size_t>("valid-freq") == 0);
  }

  bool saving() { return (batches % options_->get<size_t>("save-freq") == 0); }

  void validate(Ptr<ExpressionGraph> graph) {
    if(batches % options_->get<size_t>("valid-freq") != 0)
      return;

    bool firstValidator = true;
    for(auto validator : validators_) {
      if(!validator)
        continue;

      size_t stalledPrev = validator->stalled();
      float value = validator->validate(graph);
      if(validator->stalled() > 0)
        LOG(valid,
            "{} : {} : {} : stalled {} times",
            batches,
            validator->type(),
            value,
            validator->stalled());
      else
        LOG(valid,
            "{} : {} : {} : new best",
            batches,
            validator->type(),
            value);

      // notify training observers if the first validator did not improve
      if(firstValidator && validator->stalled() > stalledPrev)
        trainState_->newStalled(validator->stalled());
      firstValidator = false;
    }
  }

  size_t stalled() {
    if(!validators_.empty())
      if(validators_[0])
        return validators_[0]->stalled();
    return 0;
  }

  void update(float cost, Ptr<data::Batch> batch) {
    costSum += cost * batch->size();
    samples += batch->size();
    samplesDisp += batch->size();
    wordsDisp += batch->words();
    batches++;

    if(batches % options_->get<size_t>("disp-freq") == 0) {
      LOG(info,
          "Ep. {} : Up. {} : Sen. {} : Cost {:.2f} : Time {} : {:.2f} words/s",
          epochs,
          batches,
          samples,
          costSum / samplesDisp,
          timer.format(2, "%ws"),
          wordsDisp / std::stof(timer.format(5, "%w")));
      timer.start();
      costSum = 0;
      wordsDisp = 0;
      samplesDisp = 0;
    }
  }

  void load(const std::string& name) {
    std::string nameYaml = name + ".yml";
    if(boost::filesystem::exists(nameYaml)) {
      YAML::Node config = YAML::LoadFile(nameYaml);
      epochs = config["progress"]["epochs"].as<size_t>();
      batches = config["progress"]["batches"].as<size_t>();
    }
  }

  void save(const std::string& name) {
    YAML::Node config = options_->get();
    config["progress"]["epochs"] = epochs;
    config["progress"]["batches"] = batches;

    std::string nameYaml = name + ".yml";
    std::ofstream fout(nameYaml);
    fout << config;
  }

  size_t numberOfBatches() { return batches; }

  void registerTrainingObserver(Ptr<TrainingObserver> observer) {
    trainState_->registerObserver(observer);
  }

  void actAfterEpoch(TrainingState& state) {
    float factor = options_->get<double>("learning-rate-decay");
    // @TODO: move this warning to different place
    if (factor > 1.0f)
      LOG(warn, "Learning rate decay factor greater than 1.0 is unusual");

    // @TODO: remove this logging
    LOG(info,
        "[afterEpoch] Learning rate: {}, stalled: {}, max stalled: {}",
        state.eta,
        state.stalled,
        state.maxStalled);

    /* The following behaviour for learning rate decaying is implemented:
     *
     * * With only the --start-decay-epoch option enabled, the learning rate
     *   is decayed after *each* epoch starting from N-th epoch.
     *
     * * With only the --start-decay-stalled option enabled, the learning rate
     *   is decayed (*once*) if the first validation metric is not improving
     *   for N consecutive validation steps.
     *
     * * With both options enabled, the learning rate is decayed after *each*
     *   epoch starting from the first epoch for which any of those two
     *   conditions is met.
     */
    if (factor > 0.0f) {
      bool decay = false;

      int startAtEpoch = options_->get<int>("start-decay-epoch");
      if(startAtEpoch && state.epoch >= startAtEpoch)
        decay = true;

      int startWhenStalled = options_->get<int>("start-decay-stalled");
      if(startAtEpoch && startWhenStalled && state.maxStalled >= startWhenStalled)
        decay = true;

      if(decay) {
        state.eta *= factor;
        LOG(info, "Decaying learning rate to {}", state.eta);
      }
    }
  }

  void actAfterStalled(TrainingState& state) {
    float factor = options_->get<double>("learning-rate-decay");
    // @TODO: move this warning to different place
    if (factor > 1.0f)
      LOG(warn, "Learning rate decay factor greater than 1.0 is unusual");

    // @TODO: remove this logging
    LOG(info,
        "[afterStalled] Learning rate: {}, stalled: {}, max stalled: {}",
        state.eta,
        state.stalled,
        state.maxStalled);

    if (factor > 0.0f) {
      int startAtEpoch = options_->get<int>("start-decay-epoch");
      int startWhenStalled = options_->get<int>("start-decay-stalled");
      if(!startAtEpoch && startWhenStalled && state.stalled >= startWhenStalled) {
        state.eta *= factor;
        LOG(info, "Decaying learning rate to {}", state.eta);
      }
    }
  }
};
}
