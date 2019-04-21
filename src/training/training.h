#pragma once

#include "common/config.h"
#include "data/batch_generator.h"
#ifndef _MSC_VER // @TODO: include SqLite in Visual Studio project
#include "data/corpus_sqlite.h"
#endif
#include "models/model_task.h"
#include "training/scheduler.h"
#include "training/validator.h"

namespace marian {

template<class ModelWrapper>
Ptr<data::BatchStats>
miniBatchFit(Ptr<Options> options, Ptr<data::CorpusBase> data,
             Ptr<Scheduler> scheduler, Ptr<IMPIWrapper> mpi) {
  // collect stats for fitting minibatches to available memory
  Ptr<data::BatchStats> stats;
  if(options->get<bool>("mini-batch-fit")) {
    LOG(info,
        "[batching] Collecting statistics for batch fitting with step size {}",
        options->get<size_t>("mini-batch-fit-step"));
    // @TODO this should receive a function object that can generate a fake batch;
    // that way vocabs would not be exposed.
    auto model = New<ModelWrapper>(options, mpi);
    model->setScheduler(scheduler); // collectStats() needs to know about dynamic MB scaling
    stats = model->collectStats(data->getVocabs());
    LOG(info, "[batching] Done. Typical MB size is {} target words",
        stats->estimateTypicalTrgWords());
  }
  return stats;
}

template <class ModelWrapper>
class Train : public ModelTask {
private:
  Ptr<Options> options_;

public:
  Train(Ptr<Options> options) : options_(options) {}

  void run() override {
    using namespace data;

    // TRAINING RUN SETUP
    Ptr<CorpusBase> dataset = prepareTrainingData(options_);
    auto trainState = New<TrainingState>(options_->get<float>("learn-rate"));
    auto scheduler = New<Scheduler>(options_, trainState);
    auto mpi = initMPI(/*multiThreaded=*/!options_->get<bool>("sync-sgd"));
    // @TODO: do we need the multiThreaded distinction in initMPI at all?

    Ptr<BatchStats> stats
      = miniBatchFit<ModelWrapper>(options_, dataset, scheduler, mpi);
    auto batchGenerator = New<CorpusBatchGenerator>(dataset, options_, stats);
    scheduler->setupValidators(dataset->getVocabs());
    scheduler->registerTrainingObserver(batchGenerator);

    auto model = New<ModelWrapper>(options_, mpi);
    model->setScheduler(scheduler);
    model->setTypicalTrgBatchWords(batchGenerator->estimateTypicalTrgBatchWords());
    // typical no. of trg wrds in batch is needed for dynamic MiniBatch scaling
    model->load(); //!! ALSO CHANGES scheduler AND trainState ON A RESUMED RUN!

    // -- main training loop
    scheduler->started();
    bool shuffle = !options_->get<bool>("no-shuffle");
    batchGenerator->restore(trainState,shuffle);
    while(scheduler->keepGoing()) {
      batchGenerator->prepare(shuffle);

      // main training loop for one epoch
      // @TODO: try to use for(auto ...)
      for(auto batchIt = std::begin(*batchGenerator);
          batchIt != std::end(*batchGenerator);
          batchIt++) {
        if (!scheduler->keepGoing())
          break;
        model->update(*batchIt);
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
    model = nullptr; // release any reference to MPI that model may hold
    finalizeMPI(std::move(mpi));
  }
};
}  // namespace marian
