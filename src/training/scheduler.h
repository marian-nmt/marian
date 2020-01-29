#pragma once

#include "common/options.h"
#include "training/training_state.h"
#include "training/validator.h"
#include "training/communicator.h"
#include "layers/loss.h"

namespace marian {

bool getSigtermFlag();
void installSignalHandlers();

class Scheduler : public TrainingObserver {
private:
  Ptr<Options> options_;
  Ptr<TrainingState> state_;
  std::vector<Ptr<ValidatorBase>> validators_;

  bool first_{true};

  timer::Timer timer_;
  timer::Timer heartBeatTimer_;

  // determine scheduled LR decay factor (--lr-decay-inv-sqrt option)
  float getScheduledLRDecayFactor(const TrainingState& state) const {
    auto args = options_->get<std::vector<std::string>>("lr-decay-inv-sqrt");
    ABORT_IF(args.empty() || args.size() > 2, "--lr-decay-inv-sqrt argument must be one or two numbers with units");
    auto decayGoogle = SchedulingParameter::parse(args[0]);
    size_t progress = state.getProgressIn(decayGoogle.unit);
    size_t start = decayGoogle.n;
    if (args.size() > 1) {
      auto decayStart = SchedulingParameter::parse(args[1]);
      ABORT_IF(decayStart && decayStart.unit != decayGoogle.unit,
               "both --lr-decay-inv-sqrt arguments must have the same unit");
      start = decayStart.n;
    }
    if (decayGoogle && progress > start) {
      progress = progress - start + decayGoogle.n; // shift so that we get 1 at progress==start
      return (float)(std::sqrt((double)decayGoogle.n / (double)progress));
    }
    else
      return 1.f;
  }

  // update current learning rate in state.eta
  // This considers
  //  - base LR (--learn-rate)
  //  - LR warm-up (--lr-warmup, --lr=warmup-start-rate)
  //  - scheduled LR decay (--lr-decay-inv-sqrt)
  //  - state-based LR decay (--lr-decay, --lr-decay-strategy)
  void updateLearningRate(TrainingState& state) const {
    float baselr = options_->get<float>("learn-rate");

    // warm-up factor
    float warmupFactor = 1.f;
    auto warmupParam = SchedulingParameter::parse(options_->get<std::string>("lr-warmup"));
    if(warmupParam) {
      ABORT_IF(state.warmupStart && state.warmupStart.unit != warmupParam.unit,
               "lr-warmup and warmup-start must have the same unit");
      auto bno = state.getProgressIn(warmupParam.unit) - state.warmupStart.n;
      warmupFactor = std::min(1.f, (float)bno / (float)warmupParam.n);
    }

    // TODO: why lr-warmup-start-rate is extracted from options_ instead of using state.warmupStart?
    float lrStart = options_->get<float>("lr-warmup-start-rate");
    baselr = lrStart + (baselr - lrStart) * warmupFactor; // linear interpolation between
                                                          // lr-warmup-start-rate to learn-rate

    // schedule-based decay factor (--lr-decay-inv-sqrt)
    float scheduledDecayFactor = getScheduledLRDecayFactor(state);
    baselr = baselr * scheduledDecayFactor;

    // factor in state-based decay and set final LR as state.eta
    state.updateEta(baselr);
  }

  std::string formatLoss(std::string lossType,
                         bool dispLabelCounts,
                         size_t batchLabels,
                         Ptr<TrainingState> state) {
    std::stringstream ss;
    ss << "Cost ";
    ss << std::setprecision(8) << std::fixed;

    // @TODO: put a single loss formatting function into loss.h and reuse here to avoid code duplication
    // @TODO: use dispLabelCounts with any display type?
    // @TODO: bugbug cost-type ce-mean-words with multi-loss-type mean divides too much in display
    if(lossType == "ce-mean-words") {
      ss << state->costSum / state->costCount;
    } else if(lossType == "ce-sum" && dispLabelCounts) {
      ss << state->costSum / state->costCount
         << " * " << utils::withCommas(state->costCount);
      if(batchLabels > 0)
         ss << " @ " << utils::withCommas(batchLabels);
      ss << " after " << utils::withCommas(state->labelsTotal);
    } else if(lossType == "ce-sum" && !dispLabelCounts) {
      ss << state->costSum / state->updatesDisp; // average over batches
    } else if(lossType == "perplexity") {
      ss << std::exp(state->costSum / state->costCount);
    } else if(lossType == "cross-entropy" || lossType == "ce-mean") { // backwards-compat, @TODO: get rid of this?
      ss << state->costSum / state->samplesDisp;
    } else {
      ABORT("Unknown loss type {}", lossType);
    }

    return ss.str();
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
      // This ramps up MB size at start, relative to progress within warm-up period.
      size_t progress = state_->getProgressIn(mbWarmup.unit); // number of updates/labels processed
      auto progressRatio = (double)progress / (double)mbWarmup.n; // where are we relatively within target warm-up period
      // if unit is labels, then account for the fact that our increment itself is not constant
      if (mbWarmup.unit == SchedulingUnit::trgLabels)
        progressRatio = std::sqrt(progressRatio);
      if (progressRatio < 1)
        ratio *= progressRatio;
    }

    // dynamic MB-size tracking with learning rate
    // As LR goes down, MB gets ramped up by the same ratio, which has been found to be safe.
    auto mbTracking = options_->get<bool>("mini-batch-track-lr");
    if (mbTracking) {
      auto lrFactor = getScheduledLRDecayFactor(*state_) * state_->factor; // (don't include lr-warmup)
      if (lrFactor != 1)
        LOG_ONCE(info, "[scheduler] Dynamic mini-batch size adjustment enabled and kicking in");
      ratio /= lrFactor;
    }
    return ratio;
  }

  Scheduler(Ptr<Options> options, Ptr<TrainingState> state)
      : options_(options), state_(state) {
    ABORT_IF(state_->factor != 1, "state.factor unexpectedly not 1 at this point??");
    updateLearningRate(*state);
    installSignalHandlers();
  }

  bool keepGoing() {

    if(getSigtermFlag()) // received signal SIGERM => exit gracefully
      return false;

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
  void finished() {
    if (getSigtermFlag())
      LOG(info, "Training interrupted (SIGTERM).");
    else
      LOG(info, "Training finished");
  }


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
    // loaded) or if validation is scheduled for another update, or when signal SIGTERM was received
    if(getSigtermFlag() // SIGTERM was received
       || state_->validated // already validated (in resumed training, for example)
       || (!state_->enteredNewPeriodOf(options_->get<std::string>("valid-freq")) && !final)) // not now
      return;

    bool firstValidator = true;
    for(auto validator : validators_) {
      if(!validator)
        continue;

      size_t stalledPrev = validator->stalled();
      float value = validator->validate(graphs, state_);
      if(validator->stalled() > 0) {
        LOG_VALID(info,
                  "Ep. {} : Up. {} : {} : {} : stalled {} times (last best: {})",
                  state_->epochs,
                  state_->batches,
                  validator->type(),
                  value,
                  validator->stalled(), validator->lastBest());
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

  void update(StaticLoss rationalLoss, Ptr<data::Batch> batch) {
    update(rationalLoss, /*numReadBatches=*/1, /*batchSize=*/batch->size(), /*batchLabels=*/batch->wordsTrg());
  }

  // @TODO: go back to function which takes batch as an argument? The current arguments make it hard
  // to choose which subbatch should be used for speed display. For sequence-classifiers it's more interesting
  // to see the source-words consumed rather than the labels.
  void update(StaticLoss rationalLoss,
              size_t numReadBatches, // number of batches read by the reader (for seeking in case of restart)
              size_t batchSize,      // total number of sentences in batch
              size_t batchLabels,    // total number of target words in batch
              Ptr<IMPIWrapper> mpi = nullptr) {
    state_->rememberPreviousProgress();  // note: epoch increases happen at the wrong place, hence
                                         // -freq parameters do not support epoch units
    state_->validated = false;

    // Since batchLabels is counted across all MPI processes, we also should temporarily
    // extrapolate cost across MPI processes, to have numbers in the right range.
    // When doing the actual log, we then aggregate across MPI processes to get the accurate number.
    if(mpi)
      rationalLoss.loss *= mpi->numMPIProcesses();

    // @BUGBUG: rationalLoss.count is float, not a count. Possible solution: make (costSum, costCount) a StaticLoss object as well
    state_->costSum      += rationalLoss.loss;   // aggregate sum cost since last display
    state_->costCount    += (size_t)rationalLoss.count; // cost gets normalized w.r.t. this in display

    state_->updatesDisp  += 1;
    state_->samplesDisp  += batchSize;
    state_->wordsDisp    += batchLabels;  //@TODO: this is wrong        // words at given input processed since last display, for speed display

    state_->samplesEpoch += batchSize;          // sentences processed in this epoch
    // @BUGBUG: rationalLoss.count is float, not a count
    state_->labelsTotal  += (size_t)rationalLoss.count; // total labels processed

    state_->newUpdate(numReadBatches);

    // reconstruct sum cost, for displaying epoch-level averages instead of minibatch-level
    auto lossType = options_->get<std::string>("cost-type");
    auto dispLabelCounts = options_->get<bool>("disp-label-counts");  // if true then show as "cost per label * number of labels"

    if(state_->enteredNewPeriodOf(options_->get<std::string>("disp-freq")) ||
       state_->batches <= options_->get<size_t>("disp-first")) {
      // if MPI then aggregate precise cost across workers
      if(mpi) {
        state_->costSum /= mpi->numMPIProcesses(); // undo the extra scaling
        mpi->allReduce(&state_->costSum, &state_->costSum, 1, MPI_FLOAT, MPI_SUM);
      }

      if(mpi && mpi->myMPIRank() != 0) {
        // skip the report on alternate worker processes
      } else if(options_->get<bool>("lr-report")) {
        LOG(info,
            "Ep. {} : Up. {} : Sen. {} : {} : Time {:.2f}s : {:.2f} words/s : L.r. {:.4e}",
            state_->epochs,
            state_->batches,
            utils::withCommas(state_->samplesEpoch),
            formatLoss(lossType, dispLabelCounts, batchLabels, state_),
            timer_.elapsed(),
            state_->wordsDisp / timer_.elapsed(),
            state_->eta);
      } else {
        LOG(info,
            "Ep. {} : Up. {} : Sen. {} : {} : Time {:.2f}s : {:.2f} words/s",
            state_->epochs,
            state_->batches,
            utils::withCommas(state_->samplesEpoch),
            formatLoss(lossType, dispLabelCounts, 0, state_), // ignore batchLabels
            timer_.elapsed(),
            state_->wordsDisp / timer_.elapsed());
      }


      timer_.start();
      state_->costSum      = 0;
      state_->costCount    = 0;

      state_->updatesDisp  = 0;
      state_->samplesDisp  = 0;
      state_->wordsDisp    = 0;
    }

    // progress heartbeat for MS-internal Philly compute cluster
    // This environment variable exists when running on the cluster.
    using namespace std::chrono;
    if((!mpi || mpi->myMPIRank() == 0) && getenv("PHILLY_JOB_ID")
       && heartBeatTimer_.elapsed<std::chrono::minutes>() >= 10) {
      printf("PROGRESS: %.2f%%\nEVALERR: %.7f%%\n",
          (double)state_->epochs,
          state_->costSum / (state_->costCount ? state_->costCount : 1) / (mpi ? mpi->numMPIProcesses() : 1));
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
      state_->costSum      = 0;
      state_->costCount    = 0;

      state_->updatesDisp  = 0;
      state_->samplesDisp  = 0;
      state_->wordsDisp    = 0;
    }

    state_->newLoad();
  }

  void save(const std::string& name) {
    // Save config options
    std::ofstream fout(name + ".yml");
    fout << options_->asYamlString();
    // Save training progress
    state_->save(name + ".progress.yml");
  }

  size_t numberOfBatches() { return state_->batches; }

  void registerTrainingObserver(Ptr<TrainingObserver> observer) {
    state_->registerObserver(observer);
  }

  void actAfterEpoch(TrainingState& state) override {
    float factor = options_->get<float>("lr-decay");

    updateLearningRate(state);

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
        updateLearningRate(state);
        LOG(info, "Decaying learning rate to {} in epoch {}", state.eta, state.epochs);

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
    float factor = options_->get<float>("lr-decay");
    state.reset = false;

    updateLearningRate(state);

    if(factor > 0.0) {
      if(options_->get<std::string>("lr-decay-strategy") == "batches") {
        size_t start = options_->get<std::vector<size_t>>("lr-decay-start").front();
        size_t freq  = options_->get<size_t>("lr-decay-freq"); // note: unlike e.g. disp-freq, this is always in batches

        if(start > 0 && freq > 0 && state.batches >= start
           && ((state.batches - start) % freq == 0)) {
          state.factor *= factor;
          updateLearningRate(state);
          LOG(info, "Decaying learning rate to {} after {} batches", state.eta, state.batches);

          state.reset = options_->get<bool>("lr-decay-reset-optimizer");
          if(state.reset)
            LOG(info, "Resetting optimizer statistics");

          if(options_->get<bool>("lr-decay-repeat-warmup")) {
            LOG(info, "Restarting learning rate warmup");
            // TODO: avoid repeating this many times and minimize calls to options_->get
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
    float factor = options_->get<float>("lr-decay");
    state.reset = false;

    updateLearningRate(state);

    if(factor > 0.0) {
      if(options_->get<std::string>("lr-decay-strategy") == "stalled") {
        size_t startStalled = options_->get<std::vector<size_t>>("lr-decay-start").front();
        if(startStalled && state.stalled && state.stalled % startStalled == 0) {
          state.factor *= factor;
          updateLearningRate(state);
          LOG(info,
              "Decaying learning rate to {} after having stalled {} time(s)",
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
