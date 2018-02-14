#pragma once

#include "common/config.h"
#include "training/training_state.h"
#include "training/validator.h"

namespace marian {

class Scheduler : public TrainingObserver {
private:
  YAML::Node progress;

  Ptr<Config> options_;
  std::vector<Ptr<ValidatorBase>> validators_;
  bool validated_{false};

  float costSum{0};
  size_t samples{0};
  size_t samplesDisp{0};
  size_t wordsDisp{0};

  bool first_{true};

  Ptr<TrainingState> state_;

  boost::timer::cpu_timer timer;

public:
  Scheduler(Ptr<Config> options, Ptr<TrainingState> state)
      : options_(options), state_(state) {}

  bool keepGoing() {
    // stop if it reached the maximum number of epochs
    int stopAfterEpochs = options_->get<size_t>("after-epochs");
    if(stopAfterEpochs > 0 && state_->epochs > stopAfterEpochs)
      return false;

    // stop if it reached the maximum number of batch updates
    int stopAfterBatches = options_->get<size_t>("after-batches");
    if(stopAfterBatches > 0 && state_->batches >= stopAfterBatches)
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

    state_->newEpoch();
    samples = 0;

    LOG(info, "Starting epoch {}", state_->epochs);
  }

  void started() { LOG(info, "Training started"); }
  void finished() { LOG(info, "Training finished"); }

  void addValidator(Ptr<ValidatorBase> validator) {
    validators_.push_back(validator);
  }

  bool validating() {
    return (!validators_.empty()
            && state_->batches % options_->get<size_t>("valid-freq") == 0);
  }

  bool saving() {
    return (state_->batches % options_->get<size_t>("save-freq") == 0);
  }

  void validate(const std::vector<Ptr<ExpressionGraph>>& graphs, bool final = false) {
    if(validated_ || (state_->batches % options_->get<size_t>("valid-freq") != 0
                      && !final))
      return;

    bool firstValidator = true;
    for(auto validator : validators_) {
      if(!validator)
        continue;

      size_t stalledPrev = validator->stalled();
      float value = validator->validate(graphs);
      if(validator->stalled() > 0)
        LOG_VALID(info,
                  "{} : {} : {} : stalled {} times",
                  state_->batches,
                  validator->type(),
                  value,
                  validator->stalled());
      else
        LOG_VALID(info,
                  "{} : {} : {} : new best",
                  state_->batches,
                  validator->type(),
                  value);

      // notify training observers if the first validator did not improve
      if(firstValidator && validator->stalled() > stalledPrev)
        state_->newStalled(validator->stalled());
      firstValidator = false;
    }

    validated_ = true;
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
    state_->newBatch();

    if(state_->batches % options_->get<size_t>("disp-freq") == 0) {
      if(options_->get<bool>("lr-report")) {
        LOG(info,
            "Ep. {} : Up. {} : Sen. {} : Cost {:.2f} : Time {} : {:.2f} "
            "words/s : L.r. {:.4e}",
            state_->epochs,
            state_->batches,
            samples,
            costSum / samplesDisp,
            timer.format(2, "%ws"),
            wordsDisp / std::stof(timer.format(5, "%w")),
            state_->eta);
      } else {
        LOG(info,
            "Ep. {} : Up. {} : Sen. {} : Cost {:.2f} : Time {} : {:.2f} "
            "words/s",
            state_->epochs,
            state_->batches,
            samples,
            costSum / samplesDisp,
            timer.format(2, "%ws"),
            wordsDisp / std::stof(timer.format(5, "%w")));
      }
      timer.start();
      costSum = 0;
      wordsDisp = 0;
      samplesDisp = 0;
    }

    validated_ = false;
  }

  void load(const std::string& name) {
    std::string nameYaml = name + ".yml";
    if(boost::filesystem::exists(nameYaml)) {
      YAML::Node config = YAML::LoadFile(nameYaml);
      state_->epochs = config["progress"]["epochs"].as<size_t>();
      state_->batches = config["progress"]["batches"].as<size_t>();
    }
  }

  void save(const std::string& name) {
    YAML::Node config = options_->get();
    config["progress"]["epochs"] = state_->epochs;
    config["progress"]["batches"] = state_->batches;

    std::string nameYaml = name + ".yml";
    std::ofstream fout(nameYaml);
    fout << config;
  }

  size_t numberOfBatches() { return state_->batches; }

  void registerTrainingObserver(Ptr<TrainingObserver> observer) {
    state_->registerObserver(observer);
  }

  float getLearningRate(TrainingState& state) {
    float baselr = options_->get<float>("learn-rate");

    float bno = state.batches - state.warmupStart;

    size_t warmup = options_->get<size_t>("lr-warmup");
    float mult1 = 1.f;
    if(warmup > 0) {
      mult1 = std::min(1.f, bno / (float)warmup);
    }

    size_t decayGoogle = options_->get<size_t>("lr-decay-inv-sqrt");
    float mult2 = 1.f;
    if(decayGoogle > 0) {
      mult2 = std::min(
          1.f, (float)(std::sqrt(decayGoogle) / std::sqrt(state.batches)));
    }

    baselr = baselr * mult1 * mult2;

    float lrStart = options_->get<float>("lr-warmup-start-rate");
    if(lrStart > 0)
      baselr = baselr - lrStart * mult1 * mult2 + lrStart * mult2;

    return baselr;
  }

  void actAfterEpoch(TrainingState& state) {
    float factor = options_->get<double>("lr-decay");

    float baselr = getLearningRate(state);
    state.eta = baselr * state.factor;

    if(factor > 0.0) {
      bool decay = false;
      auto strategy = options_->get<std::string>("lr-decay-strategy");
      state.reset = false;

      if(strategy == "epoch" || strategy == "epoch+batches"
         || strategy == "epoch+stalled") {
        int startEpoch
            = options_->get<std::vector<size_t>>("lr-decay-start").front();
        if(startEpoch && state.epochs >= startEpoch)
          decay = true;
      }

      if(strategy == "epoch+batches") {
        int startBatches
            = options_->get<std::vector<size_t>>("lr-decay-start")[1];
        if(startBatches && state.batches >= startBatches)
          decay = true;
      }
      if(strategy == "epoch+stalled") {
        int startStalled
            = options_->get<std::vector<size_t>>("lr-decay-start")[1];
        if(startStalled && state.maxStalled >= startStalled)
          decay = true;
      }

      if(decay) {
        state.factor *= factor;
        state.eta = baselr * state.factor;
        LOG(info,
            "Decaying learning rate to {} in epoch {}",
            state.eta,
            state.epochs);

        state.reset = options_->get<bool>("lr-decay-reset-optimizer");
        if(state.reset)
          LOG(info, "Resetting optimizer statistics");

        if(options_->get<bool>("lr-decay-repeat-warmup")) {
          LOG(info, "Restarting learning rate warmup");
          state.warmupStart = state.batches;
        }
      }
    }
  }

  void actAfterBatches(TrainingState& state) {
    float factor = options_->get<double>("lr-decay");
    state.reset = false;

    float baselr = getLearningRate(state);
    state.eta = baselr * state.factor;

    if(factor > 0.0) {
      if("batches" == options_->get<std::string>("lr-decay-strategy")) {
        int start
            = options_->get<std::vector<size_t>>("lr-decay-start").front();
        int freq = options_->get<size_t>("lr-decay-freq");

        if(start > 0 && freq > 0 && state.batches >= start
           && ((state.batches - start) % freq == 0)) {
          state.factor *= factor;
          state.eta = baselr * state.factor;
          LOG(info,
              "Decaying learning rate to {} after {} batches",
              state.eta,
              state.batches);

          state.reset = options_->get<bool>("lr-decay-reset-optimizer");
          if(state.reset)
            LOG(info, "Resetting optimizer statistics");

          if(options_->get<bool>("lr-decay-repeat-warmup")) {
            LOG(info, "Restarting learning rate warmup");
            state.warmupStart = state.batches;
          }
        }
      }
    }

    if(first_ && options_->get<bool>("lr-warmup-at-reload")) {
      LOG(info, "Restarting learning rate warmup");
      state.warmupStart = state.batches;
    }

    if(options_->get<bool>("lr-warmup-cycle")) {
      size_t warmup = options_->get<size_t>("lr-warmup");
      if(warmup > 0 && state.batches % warmup == 0)
        state.warmupStart = state.batches;
    }

    first_ = false;
  }

  void actAfterStalled(TrainingState& state) {
    float factor = options_->get<double>("lr-decay");
    state.reset = false;

    float baselr = getLearningRate(state);
    state.eta = baselr * state.factor;

    if(factor > 0.0) {
      if(options_->get<std::string>("lr-decay-strategy") == "stalled") {
        int startStalled
            = options_->get<std::vector<size_t>>("lr-decay-start").front();
        if(startStalled && state.stalled && state.stalled % startStalled == 0) {
          state.factor *= factor;
          state.eta = baselr * state.factor;
          LOG(info,
              "Decaying learning rate to {} after stalled {} time(s)",
              state.eta,
              state.stalled);

          state.reset = options_->get<bool>("lr-decay-reset-optimizer");
          if(state.reset)
            LOG(info, "Resetting optimizer statistics");

          if(options_->get<bool>("lr-decay-repeat-warmup")) {
            LOG(info, "Restarting learning rate warmup");
            state.warmupStart = state.batches;
          }
        }
      }
    }
  }
};
}
