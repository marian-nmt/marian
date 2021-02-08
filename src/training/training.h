#pragma once

#include "common/config.h"
#include "common/utils.h"
#include "data/batch_generator.h"
#ifndef _MSC_VER // @TODO: include SqLite in Visual Studio project
#include "data/corpus_sqlite.h"
#endif
#include "models/model_task.h"
#include "training/scheduler.h"
#include "training/validator.h"

namespace marian {

template <class ModelWrapper>
class Train : public ModelTask {
private:
  Ptr<Options> options_;
  void installCustomSignalHandlers();

public:
  Train(Ptr<Options> options) : options_(options) {}

  void run() override {
    using namespace data;
    
    // MPI init should be first thing in training
    auto mpi = initMPI(/*multiThreaded=*/!options_->get<bool>("sync-sgd")); // @TODO: do we need the multiThreaded distinction at all?
    
    if(mpi) { // if we run MPI, then make sure to sync seed across processes as first action
      mpi->bCast(&Config::seed, 1, IMPIWrapper::getDataType(&Config::seed));
      LOG(info, "Synced seed {}", Config::seed);
    }

    Ptr<CorpusBase> dataset;
    auto corpusSeed = Config::seed + (mpi ? mpi->myMPIRank() : 0); // @BUGBUG: no correct resume right now
    if(!options_->get<std::string>("sqlite").empty())
#ifndef _MSC_VER // @TODO: include SqLite in Visual Studio project
      dataset = New<CorpusSQLite>(options_, /*translate=*/false, corpusSeed);
#else
      ABORT("SqLite presently not supported on Windows");
#endif
    else
      dataset = New<Corpus>(options_, /*translate=*/false, corpusSeed);

    dataset->prepare();

    Ptr<BatchStats> stats;
    if(options_->get<bool>("mini-batch-fit")) {
      LOG(info,
          "[batching] Collecting statistics for batch fitting with step size {}",
          options_->get<size_t>("mini-batch-fit-step"));
      // @TODO this should receive a function object that can generate a fake batch;
      // that way vocabs would not be exposed.
      auto model = New<ModelWrapper>(options_, mpi);

      // use temporary scheduler to make sure everything gets destroyed properly
      // otherwise the scheduler believes that registered objects still exist
      auto tempTrainState = New<TrainingState>(options_->get<float>("learn-rate"));
      auto tempScheduler = New<Scheduler>(options_, tempTrainState, mpi);

      model->setScheduler(tempScheduler); // collectStats() needs to know about dynamic MB scaling
      stats = model->collectStats(dataset->getVocabs());
      LOG(info, "[batching] Done. Typical MB size is {} target words", utils::withCommas(stats->estimateTypicalTrgWords()));
    }

    auto trainState = New<TrainingState>(options_->get<float>("learn-rate"));
    auto scheduler = New<Scheduler>(options_, trainState, mpi);

    if((options_->hasAndNotEmpty("valid-sets") || options_->hasAndNotEmpty("valid-script-path"))
       && SchedulingParameter::parse(options_->get<std::string>("valid-freq"))) {
      for(auto validator : Validators(dataset->getVocabs(), options_))
        scheduler->addValidator(validator);
    }

    auto batchGenerator = New<CorpusBatchGenerator>(dataset, options_, stats);

    scheduler->registerTrainingObserver(batchGenerator);

    auto model = New<ModelWrapper>(options_, mpi);
    model->setScheduler(scheduler);
    model->setTypicalTrgBatchWords(batchGenerator->estimateTypicalTrgBatchWords()); // needed for dynamic MB scaling
    model->load();

    bool restored = !options_->get<bool>("no-restore-corpus")
                    && batchGenerator->restore(trainState);

    // We only want custom behavior once training starts.
    installCustomSignalHandlers();

    // -- main training loop
    scheduler->started();
    while(scheduler->keepGoing()) {
      if(!restored)
        batchGenerator->prepare();
      restored = false;

      // main training loop for one epoch
      for(auto batch : *batchGenerator) {
        if (!scheduler->keepGoing())
          break;
        model->update(batch);
      }

      if(scheduler->keepGoing())
        scheduler->increaseEpoch();
    }
    scheduler->finished();

    model->finalize(); // allow async to sync before final save   --@TODO: rename, or move into save()

    // Avoid saving the model twice if it has been loaded and training did not progress
    if(!trainState->loaded)
      model->save(true);

    // Signal success to a potential MPI runner
    model = nullptr;     // release any reference to MPI that model may hold
    scheduler = nullptr; // as above
    finalizeMPI(std::move(mpi));
  }
};

template <class ModelWrapper>
void Train<ModelWrapper>::installCustomSignalHandlers(){
  const std::string sigTermAction = options_->get<std::string>("sigterm");
  if (sigTermAction == "save-and-exit") {
    LOG(debug, "Will save before exiting upon SIGTERM.");
    signal(SIGTERM, requestSaveAndExit);
  }
  else if (sigTermAction != "exit-immediately")
    ABORT("Unrecognized value '{}' for --sigterm", sigTermAction);
}

}  // namespace marian
