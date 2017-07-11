#pragma once

#include "common/config.h"
#include "training/training_state.h"
#include "training/validator.h"

namespace marian {

template <class DataSet>
class Scheduler : public TrainingObserver {
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
  Scheduler(Ptr<Config> options, Ptr<TrainingState> state)
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
    LOG(info)->info("Seen {} samples", samples);

    epochs++;
    trainState_->newEpoch(epochs);
    samples = 0;

    LOG(info)->info("Starting epoch {}", epochs);
  }

  void started() { LOG(info)->info("Training started"); }
  void finished() { LOG(info)->info("Training finshed"); }

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
        LOG(valid)
            ->info("{} : {} : {} : stalled {} times",
                   batches,
                   validator->type(),
                   value,
                   validator->stalled());
      else
        LOG(valid)
            ->info(
                "{} : {} : {} : new best", batches, validator->type(), value);

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
      LOG(info)
          ->info(
              "Ep. {} : Up. {} : Sen. {} : Cost {:.2f} : Time {} : {:.2f} "
              "words/s",
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

    trainState_->newBatches(batches);
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
    float factor = options_->get<double>("lr-decay");
    if(factor > 0.0) {
      bool decay = false;
      auto strategy = options_->get<std::string>("lr-decay-strategy");
      int startEpoch
          = options_->get<std::vector<size_t>>("lr-decay-start").front();

      if(strategy == "epoch") {
        if(startEpoch && state.epochs >= startEpoch)
          decay = true;
      }
      if(strategy == "epoch+batches") {
        int startBatches
            = options_->get<std::vector<size_t>>("lr-decay-start")[1];
        if(startEpoch && startBatches && state.batches >= startBatches)
          decay = true;
      }
      if(strategy == "epoch+stalled") {
        int startStalled
            = options_->get<std::vector<size_t>>("lr-decay-start")[1];
        if(startEpoch && startStalled && state.maxStalled >= startStalled)
          decay = true;
      }

      if(decay) {
        state.eta *= factor;
        LOG(info)
            ->info("Decaying learning rate to {} in epoch {}",
                   state.eta,
                   state.epochs);
      }
    }
  }

  void actAfterBatches(TrainingState& state) {
    float factor = options_->get<double>("lr-decay");
    if(factor > 0.0) {
      if("batches" == options_->get<std::string>("lr-decay-strategy")) {
        int start
            = options_->get<std::vector<size_t>>("lr-decay-start").front();
        int freq = options_->get<size_t>("lr-decay-freq");

        if(start > 0 && freq > 0 && state.batches >= start
           && ((state.batches - start) % freq == 0)) {
          state.eta *= factor;
          LOG(info)
              ->info("Decaying learning rate to {} after {} batches",
                     state.eta,
                     state.batches);
        }
      }
    }
  }

  void actAfterStalled(TrainingState& state) {
    float factor = options_->get<double>("lr-decay");
    if(factor > 0.0) {
      if("stalled" == options_->get<std::string>("lr-decay-strategy")) {
        int startStalled
            = options_->get<std::vector<size_t>>("lr-decay-start").front();
        if(startStalled && state.stalled >= startStalled) {
          state.eta *= factor;
          LOG(info)
              ->info("Decaying learning rate to {} after stalled {} time(s)",
                     state.eta,
                     state.stalled);
        }
      }
    }
  }
};
}
