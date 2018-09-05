#pragma once

#include "common/config.h"
#include "training/training_state.h"
#include "training/validator.h"

namespace marian {

class Scheduler : public TrainingObserver {
private:
  Ptr<Config> options_;
  std::vector<Ptr<ValidatorBase>> validators_;

  bool first_{true};

  Ptr<TrainingState> state_;

  boost::timer::cpu_timer timer;

  float getLearningRate(TrainingState& state) {
    float baselr = options_->get<float>("learn-rate");

    auto bno = state.batches - state.warmupStart;

    size_t warmup = options_->get<size_t>("lr-warmup");
    float mult1 = 1.f;
    if(warmup > 0) {
      mult1 = std::min(1.f, (float)bno / (float)warmup);
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

public:
  Scheduler(Ptr<Config> options, Ptr<TrainingState> state)
      : options_(options), state_(state) {
    state_->eta = getLearningRate(*state);
  }

  bool keepGoing() {
    // stop if it reached the maximum number of epochs
    size_t stopAfterEpochs = options_->get<size_t>("after-epochs");
    if(stopAfterEpochs > 0 && state_->epochs > stopAfterEpochs)
      return false;

    // stop if it reached the maximum number of batch updates
    size_t stopAfterBatches = options_->get<size_t>("after-batches");
    if(stopAfterBatches > 0 && state_->batches >= stopAfterBatches)
      return false;

    // stop if the first validator did not improve for a given number of checks
    size_t stopAfterStalled = options_->get<size_t>("early-stopping");
    if(stopAfterStalled > 0 && !validators_.empty()
       && stalled() >= stopAfterStalled)
      return false;

    return true;
  }

  void increaseEpoch() {
    LOG(info, "Seen {} samples", state_->samplesEpoch);
    state_->newEpoch();
    LOG(info, "Starting epoch {}", state_->epochs);
  }

  void started() { LOG(info, "Training started"); }
  void finished() { LOG(info, "Training finished"); }

  void addValidator(Ptr<ValidatorBase> validator) {
    validators_.push_back(validator);

    registerTrainingObserver(validators_.back());
    if(!state_->loaded) {
      state_->validators[validator->type()]["last-best"]
          = validator->initScore();
      state_->validators[validator->type()]["stalled"] = 0;
    }
    if(validators_.size() == 1)
      state_->validator = validator->type();
  }

  bool validating() {
    return (!validators_.empty()
            && state_->batches % options_->get<size_t>("valid-freq") == 0);
  }

  bool saving() {
    return (state_->batches % options_->get<size_t>("save-freq") == 0);
  }

  void validate(const std::vector<Ptr<ExpressionGraph>>& graphs,
                bool final = false) {
    // Do not validate if already validated (for instance, after the model is
    // loaded) or if validation is scheduled for another update
    if(state_->validated
       || (state_->batches % options_->get<size_t>("valid-freq") != 0
           && !final))
      return;

    bool firstValidator = true;
    for(auto validator : validators_) {
      if(!validator)
        continue;

      size_t stalledPrev = validator->stalled();
      float value = validator->validate(graphs);
      if(validator->stalled() > 0) {
        LOG_VALID(info,
                  "Ep. {} : Up. {} : {} : {} : stalled {} times",
                  state_->epochs,
                  state_->batches,
                  validator->type(),
                  value,
                  validator->stalled());
      } else {
        LOG_VALID(info,
                  "Ep. {} : Up. {} : {} : {} : new best",
                  state_->epochs,
                  state_->batches,
                  validator->type(),
                  value);

        if(firstValidator)
          state_->validBest = value;
      }

      state_->validators[validator->type()]["last-best"]
          = validator->lastBest();
      state_->validators[validator->type()]["stalled"] = validator->stalled();

      // notify training observers if the first validator did not improve
      if(firstValidator && validator->stalled() > stalledPrev)
        state_->newStalled(validator->stalled());
      firstValidator = false;
    }

    state_->validated = true;
  }

  size_t stalled() {
    if(!validators_.empty())
      if(validators_[0])
        return validators_[0]->stalled();
    return 0;
  }

  void update(float cost, Ptr<data::Batch> batch) {
    update(cost, std::vector<Ptr<data::Batch>>({batch}));
  }

  void update(float cost, const std::vector<Ptr<data::Batch>>& batches) {
    state_->validated = false;

    size_t batchSize = 0;    // number of sentences in batch
    size_t batchLabels = 0;  // number of target words in batch

    for(const auto& batch : batches) {
      batchSize += batch->size();
      batchLabels += batch->words(-1);
    }

    // reconstruct sum cost, for displaying epoch-level averages instead of
    // minibatch-level
    auto costType = options_->get<std::string>("cost-type");
    auto dispLabelCounts = options_->get<bool>(
        "disp-label-counts");  // if true then show as "cost per label * number
                               // of labels"
    if(dispLabelCounts) {
      auto count =  // what was cost normalized with originally?
          /*if*/ (costType == "ce-sum")
              ? 1
              /*else if*/
              : ((costType == "ce-mean-words")
                     ? batchLabels
                     /*else*/
                     :  // all others: treat like ce-mean (not correct for some)
                     batchSize);
      state_->costSum += cost * count;  // aggregate sum cost since last display
      state_->costCount
          += batchLabels;  // cost gets normalized w.r.t. this in display
    } else {               // (back compat)
      state_->costSum += cost * batchSize;
      state_->costCount += batchSize;
    }
    state_->wordsDisp += batchLabels;    // target words processed since last
                                         // display, for speed display
    state_->samplesEpoch += batchSize;   // sentences processed in this epoch
    state_->labelsTotal += batchLabels;  // total labels processed

    state_->newBatch();

    if(state_->batches % options_->get<size_t>("disp-freq") == 0) {
      if(dispLabelCounts) {
        if(options_->get<bool>(
               "lr-report")) {  // if true then show the learning rate
          LOG(info,
              // TODO: change Cost back to {:.2f}
              "Ep. {} : Up. {} : Sen. {} : Cost {:.8f} * {} after {} : Time {} "
              ": {:.2f} "
              "words/s : L.r. {:.4e}",
              state_->epochs,
              state_->batches,
              state_->samplesEpoch,
              state_->costSum / state_->costCount,
              state_->costCount,  // show cost as "av * count"
              state_->labelsTotal,
              timer.format(2, "%ws"),
              state_->wordsDisp / std::stof(timer.format(5, "%w")),
              state_->eta);
        } else {
          LOG(info,
              "Ep. {} : Up. {} : Sen. {} : Cost {:.8f} * {} after {} : Time {} "
              ": {:.2f} "
              "words/s",
              state_->epochs,
              state_->batches,
              state_->samplesEpoch,
              state_->costSum / state_->costCount,
              state_->costCount,
              state_->labelsTotal,
              timer.format(2, "%ws"),
              state_->wordsDisp / std::stof(timer.format(5, "%w")));
        }
      } else {
        if(options_->get<bool>("lr-report")) {
          LOG(info,
              "Ep. {} : Up. {} : Sen. {} : Cost {:.2f} : Time {} : {:.2f} "
              "words/s : L.r. {:.4e}",
              state_->epochs,
              state_->batches,
              state_->samplesEpoch,
              state_->costSum / state_->costCount,
              timer.format(2, "%ws"),
              state_->wordsDisp / std::stof(timer.format(5, "%w")),
              state_->eta);
        } else {
          LOG(info,
              "Ep. {} : Up. {} : Sen. {} : Cost {:.2f} : Time {} : {:.2f} "
              "words/s",
              state_->epochs,
              state_->batches,
              state_->samplesEpoch,
              state_->costSum / state_->costCount,
              timer.format(2, "%ws"),
              state_->wordsDisp / std::stof(timer.format(5, "%w")));
        }
      }
      // progress heartbeat for MS-internal Philly compute cluster
      if(getenv("PHILLY_JOB_ID"))  // this environment variable exists when
                                   // running on the cluster
        printf("PROGRESS: %.2f%%\nEVALERR: %.7f\n",
               (double)state_->epochs,
               state_->costSum / state_->costCount),
            fflush(stdout);
      timer.start();
      state_->costSum = 0;
      state_->costCount = 0;
      state_->wordsDisp = 0;
    }
  }

  void load(const std::string& name) {
    std::string nameYaml = name + ".progress.yml";
    if(boost::filesystem::exists(nameYaml))
      state_->load(nameYaml);

    if(options_->get<bool>("no-restore-corpus")) {
      state_->samplesEpoch = 0;
      state_->costSum = 0;
      state_->costCount = 0;
      state_->wordsDisp = 0;
    }

    state_->newLoad();
  }

  void save(const std::string& name) {
    // Save config options
    YAML::Node config = options_->get();
    std::ofstream fout(name + ".yml");
    fout << config;
    // Save training progress
    state_->save(name + ".progress.yml");
  }

  size_t numberOfBatches() { return state_->batches; }

  void registerTrainingObserver(Ptr<TrainingObserver> observer) {
    state_->registerObserver(observer);
  }

  void actAfterEpoch(TrainingState& state) override {
    float factor = (float)options_->get<double>("lr-decay"); // @TODO: <float>?

    float baselr = getLearningRate(state);
    state.eta = baselr * state.factor;

    if(factor > 0.0) {
      bool decay = false;
      auto strategy = options_->get<std::string>("lr-decay-strategy");
      state.reset = false;

      if(strategy == "epoch" || strategy == "epoch+batches"
         || strategy == "epoch+stalled") {
        size_t startEpoch
            = options_->get<std::vector<size_t>>("lr-decay-start").front();
        if(startEpoch && state.epochs >= startEpoch)
          decay = true;
      }

      if(strategy == "epoch+batches") {
        size_t startBatches
            = options_->get<std::vector<size_t>>("lr-decay-start")[1];
        if(startBatches && state.batches >= startBatches)
          decay = true;
      }
      if(strategy == "epoch+stalled") {
        size_t startStalled
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

  void actAfterBatches(TrainingState& state) override {
    float factor = (float)options_->get<double>("lr-decay"); // @TODO: <float>?
    state.reset = false;

    float baselr = getLearningRate(state);
    state.eta = baselr * state.factor;

    if(factor > 0.0) {
      if("batches" == options_->get<std::string>("lr-decay-strategy")) {
        size_t start
            = options_->get<std::vector<size_t>>("lr-decay-start").front();
        size_t freq = options_->get<size_t>("lr-decay-freq");

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

  void actAfterStalled(TrainingState& state) override {
    float factor = (float)options_->get<double>("lr-decay"); // @TODO: <float>?
    state.reset = false;

    float baselr = getLearningRate(state);
    state.eta = baselr * state.factor;

    if(factor > 0.0) {
      if(options_->get<std::string>("lr-decay-strategy") == "stalled") {
        size_t startStalled
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
}  // namespace marian
