#pragma once

#include "common/options.h"
#include "training/training_state.h"
#include "training/validator.h"
#include "training/communicator.h"

namespace marian {

class Scheduler : public TrainingObserver {
private:
  Ptr<Options> options_;
  std::vector<Ptr<ValidatorBase>> validators_;

  bool first_{true};

  Ptr<TrainingState> state_;

  timer::Timer timer_, heartBeatTimer_;

  // determine LR decay factor from --lr-decay-inv-sqrt option
  float getLearningRateDecayFactor(const TrainingState& state) const {
    auto args = options_->get<std::vector<std::string>>("lr-decay-inv-sqrt");
    ABORT_IF(args.empty() || args.size() > 2, "--lr-decay-inv-sqrt argument must be one or two numbers with units");
    auto decayGoogle = SchedulingParameter::parse(args[0]);
    size_t progress = state.getProgressIn(decayGoogle.unit);
    size_t start = decayGoogle.n;
    if (args.size() > 1) {
      auto decayStart = SchedulingParameter::parse(args[1]);
      ABORT_IF(decayStart && decayStart.unit != decayGoogle.unit, "both --lr-decay-inv-sqrt arguments must have the same unit");
      start = decayStart.n;
    }
    if (decayGoogle && progress > start) {
      progress = progress - start + decayGoogle.n; // shift so that we get 1 at progress==start
      return (float)(std::sqrt((double)decayGoogle.n / (double)progress));
    }
    else
      return 1.f;
  }

  // determine the dynamically adjusted learning rate, incl. warm-up and decay
  float getLearningRate(const TrainingState& state) const {
    float baselr = options_->get<float>("learn-rate");

    float mult1 = 1.f;
    auto warmup = SchedulingParameter::parse(options_->get<std::string>("lr-warmup"));
    if(warmup) {
      ABORT_IF(state.warmupStart && state.warmupStart.unit != warmup.unit, "lr-warmup and warmup-start must have the same unit");
      auto bno = state.getProgressIn(warmup.unit) - state.warmupStart.n;
      mult1 = std::min(1.f, (float)bno / (float)warmup.n);
    }

    float mult2 = getLearningRateDecayFactor(state);

    baselr = baselr * mult1 * mult2;

    float lrStart = options_->get<float>("lr-warmup-start-rate");
    if(lrStart > 0)
      baselr = baselr - lrStart * mult1 * mult2 + lrStart * mult2;

    return baselr;
  }

public:
  // test if any parameters specify dynamic MB scaling
  bool isDynamicMBSizeScaling() const {
    auto mbWarmup = SchedulingParameter::parse(options_->get<std::string>("mini-batch-warmup"));
    auto mbTracking = options_->get<bool>("mini-batch-track-lr");
    return mbWarmup || mbTracking;
  }

  // determine dynamic MB scaling factor
  double getDynamicMBSizeMultiplier() const {
    double ratio = 1.0;

    auto mbWarmup = SchedulingParameter::parse(options_->get<std::string>("mini-batch-warmup"));
    if (mbWarmup) {
      // mini-batch-warmup
      LOG_ONCE(info, "[scheduler] Mini-batch size warmup {}", std::string(mbWarmup));
      // This scales MB size up from the start, relative to progress within warm-up period.
      size_t progress = state_->getProgressIn(mbWarmup.unit); // number of updates/labels processed
      auto progressRatio = (double)progress / (double)mbWarmup.n; // where are we relatively within target warm-up period
      // if unit is labels, then account for the fact that our increment itself is not constant
      if (mbWarmup.unit == SchedulingUnit::trgLabels)
        progressRatio = std::sqrt(progressRatio);
      // apply ratio to actual batch size
      ratio *= progressRatio;
    }

    // dynamic MB-size tracking with learning rate
    // As LR goes down, MB gets ramped up by the same ratio, which has been found to be safe.
    auto mbTracking = options_->get<bool>("mini-batch-track-lr");
    if (mbTracking) {
      auto lrFactor = getLearningRateDecayFactor(*state_);
      if (lrFactor != 1)
        LOG_ONCE(info, "[scheduler] Dynamic mini-batch size adjustment enabled and kicking in");
      ratio /= lrFactor;
    }
    return ratio;
  }

  Scheduler(Ptr<Options> options, Ptr<TrainingState> state)
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
            && state_->enteredNewPeriodOf(options_->get<std::string>("valid-freq"))
            && keepGoing());
  }

  bool saving() {
    return state_->enteredNewPeriodOf(options_->get<std::string>("save-freq"));
  }

  void validate(const std::vector<Ptr<ExpressionGraph>>& graphs,
                bool final = false) {
    // Do not validate if already validated (for instance, after the model is
    // loaded) or if validation is scheduled for another update
    if(state_->validated
       || (!state_->enteredNewPeriodOf(options_->get<std::string>("valid-freq"))
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
                  "Ep. {} : Up. {} : {} : {} : stalled {} times",  /*"(last best: {})",*/ // @TODO (LOGGING CHANGE)
                  state_->epochs,
                  state_->batches,
                  validator->type(),
                  value,
                  validator->stalled() /*, validator->lastBest()*/); // @TODO (LOGGING CHANGE)
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
    update(cost,
        /*numReadBatches=*/1, /*batchSize=*/batch->size(), /*batchLabels=*/batch->wordsTrg());
  }

  void update(float cost,
              size_t numReadBatches, // number of batches read by the reader (for seeking in case of restart)
              size_t batchSize,      // total number of sentences in batch
              size_t batchLabels,    // total number of target words in batch
              Ptr<IMPIWrapper> mpi = nullptr) {
    state_->rememberPreviousProgress(); // note: epoch increases happen at the wrong place, hence -freq parameters do not support epoch units
    state_->validated = false;

    // Since batchLabels is counted across all MPI processes, we also should temporarily
    // extrapolate cost across MPI processes, to have numbers in the right range.
    // When doing the actual log, we then aggregate across MPI processes to get the accurate number.
    if (mpi)
      cost *= mpi->numMPIProcesses(); // @BUGBUG: this is presently correct for ce-sum, but possibly not the av-based losses

    // reconstruct sum cost, for displaying epoch-level averages instead of minibatch-level
    auto costType = options_->get<std::string>("cost-type");
    auto dispLabelCounts = options_->get<bool>(
        "disp-label-counts");  // if true then show as "cost per label * number of labels"
    if(dispLabelCounts) {
      auto count =  // what was cost normalized with originally?
          /*if*/ (costType == "ce-sum") ?
            1
          /*else if*/ : ((costType == "ce-mean-words") ?
            batchLabels
          /*else*/ :  // all others: treat like ce-mean (not correct for some)
            batchSize);
      state_->costSum   += cost * count; // aggregate sum cost since last display
      state_->costCount += batchLabels;  // cost gets normalized w.r.t. this in display
    } else {               // (back compat)
      state_->costSum   += cost * batchSize;
      state_->costCount += batchSize;
    }
    state_->wordsDisp    += batchLabels; // target words processed since last display, for speed display
    state_->samplesEpoch += batchSize;   // sentences processed in this epoch
    state_->labelsTotal  += batchLabels; // total labels processed

    state_->newUpdate(numReadBatches);

    if(state_->enteredNewPeriodOf(options_->get<std::string>("disp-freq")) ||
       state_->batches <= options_->get<size_t>("disp-first")) {
      // if MPI then aggregate precise cost across workers
      if (mpi) {
        state_->costSum /= mpi->numMPIProcesses(); // undo the extra scaling
        mpi->allReduce(&state_->costSum, &state_->costSum, 1, MPI_FLOAT, MPI_SUM);
      }
      if (mpi && mpi->myMPIRank() != 0)
        ; // skip the report on alternate worker processes
      else if(dispLabelCounts) {
        if(options_->get<bool>("lr-report")) {  // if true then show the learning rate
          LOG(info,
              "Ep. {} : Up. {} : Sen. {} : Cost {:.8f} * {} @ {} after {} : Time {:.2f}s : {:.2f} "
              "words/s : L.r. {:.4e}",
              state_->epochs,
              state_->batches,
              utils::withCommas(state_->samplesEpoch),
              state_->costSum / state_->costCount,
              utils::withCommas(state_->costCount),  // show cost as "av * count"
              batchLabels,
              utils::withCommas(state_->labelsTotal),
              timer_.elapsed(),
              state_->wordsDisp / timer_.elapsed(),
              state_->eta);
        } else {
          LOG(info,
              "Ep. {} : Up. {} : Sen. {} : Cost {:.8f} * {} @ {} after {} : Time {:.2f}s : {:.2f} "
              "words/s",
              state_->epochs,
              state_->batches,
              utils::withCommas(state_->samplesEpoch),
              state_->costSum / state_->costCount,
              utils::withCommas(state_->costCount),
              batchLabels,
              utils::withCommas(state_->labelsTotal),
              timer_.elapsed(),
              state_->wordsDisp / timer_.elapsed());
        }
      } else {
        if(options_->get<bool>("lr-report")) {
          LOG(info,
              "Ep. {} : Up. {} : Sen. {} : Cost {:.8f} : Time {:.2f}s : {:.2f} words/s : L.r. {:.4e}",
              state_->epochs,
              state_->batches,
              utils::withCommas(state_->samplesEpoch),
              state_->costSum / state_->costCount,
              timer_.elapsed(),
              state_->wordsDisp / timer_.elapsed(),
              state_->eta);
        } else {
          LOG(info,
              "Ep. {} : Up. {} : Sen. {} : Cost {:.8f} : Time {:.2f}s : {:.2f} words/s",
              state_->epochs,
              state_->batches,
              utils::withCommas(state_->samplesEpoch),
              state_->costSum / state_->costCount,
              timer_.elapsed(),
              state_->wordsDisp / timer_.elapsed());
        }
      }
      timer_.start();
      state_->costSum = 0;
      state_->costCount = 0;
      state_->wordsDisp = 0;
    }
    // progress heartbeat for MS-internal Philly compute cluster
    // This environment variable exists when running on the cluster.
    using namespace std::chrono;
    if((!mpi || mpi->myMPIRank() == 0) && getenv("PHILLY_JOB_ID")
       && heartBeatTimer_.elapsed<std::chrono::minutes>() >= 10) {
      printf("PROGRESS: %.2f%%\nEVALERR: %.7f%%\n",
          (double)state_->epochs,
          state_->costSum / state_->costCount / (mpi ? mpi->numMPIProcesses() : 1));
      fflush(stdout);
      std::cout << "MBSIZE: " << batchLabels << " after " << state_->batches << " updates = " << state_->labelsTotal << " labels" << std::endl << std::flush;
      heartBeatTimer_.start();
    }
  }

  void load(const std::string& name) {
    std::string nameYaml = name + ".progress.yml";
    if(filesystem::exists(nameYaml))
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
    YAML::Node yaml = options_->getYaml();
    std::ofstream fout(name + ".yml");
    fout << yaml;
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
          state.warmupStart.n = state.getProgressIn(SchedulingParameter::parse(options_->get<std::string>("lr-warmup")).unit);
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
      if(options_->get<std::string>("lr-decay-strategy") == "batches") {
        size_t start = options_->get<std::vector<size_t>>("lr-decay-start").front();
        size_t freq  = options_->get<size_t>("lr-decay-freq"); // note: unlike e.g. disp-freq, this is always in batches

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
            state.warmupStart.n = state.getProgressIn(SchedulingParameter::parse(options_->get<std::string>("lr-warmup")).unit);
          }
        }
      }
    }

    if(first_ && options_->get<bool>("lr-warmup-at-reload")) {
      LOG(info, "Restarting learning rate warmup");
      state.warmupStart.n = state.getProgressIn(SchedulingParameter::parse(options_->get<std::string>("lr-warmup")).unit);
    }

    if(options_->get<bool>("lr-warmup-cycle")) {
      if(state_->enteredNewPeriodOf(options_->get<std::string>("lr-warmup")))
        state.warmupStart.n = state.getProgressIn(SchedulingParameter::parse(options_->get<std::string>("lr-warmup")).unit);
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
            state.warmupStart.n = state.getProgressIn(SchedulingParameter::parse(options_->get<std::string>("lr-warmup")).unit);
          }
        }
      }
    }
  }
};
}  // namespace marian
