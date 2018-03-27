#include "training/graph_group_singleton.h"
#include "functional/functional.h"
#include "tensors/tensor_operators.h"

namespace marian {

void SingletonGraph::setScheduler(Ptr<Scheduler> scheduler) {
  scheduler_ = scheduler;
  // optimizer has to be registered last to see changes of learning rate
  scheduler_->registerTrainingObserver(scheduler_);
  scheduler_->registerTrainingObserver(opt_);
}

void SingletonGraph::updateMovingAverage(Tensor mvAvgParams,
                                         Tensor params,
                                         size_t batches) {
  using namespace functional;
  float decay
      = std::max(mvDecay_, 1.f - (float)(batches + 1) / (float)(batches + 10));
  Element(_1 = ((1.f - decay) * _1) + (decay * _2), mvAvgParams, params);
}

void SingletonGraph::execute(Ptr<data::Batch> batch) {
  auto costNode = builder_->build(graph_, batch);

  graph_->forward();
  float cost = costNode->scalar();
  graph_->backward();

  // Get batch stats
  size_t batch_words = batch->wordsTrg();

  if(scaleLearningRate_) {
    opt_->update(graph_, batch_words / avgBatchWords_);
  } else {
    opt_->update(graph_);
  }

  if(mvAvg_) {
    ABORT_IF(!scheduler_, "Scheduler is required for exponential smoothing");

    if(!mvAvgGraph_) {
      mvAvgGraph_ = New<ExpressionGraph>();
      mvAvgGraph_->setDevice(graph_->getDevice());
      mvAvgGraph_->copyParams(graph_);
    } else {
      updateMovingAverage(mvAvgGraph_->params()->vals(),
                          graph_->params()->vals(),
                          scheduler_->numberOfBatches());
    }
  }

  if(scheduler_) {
    scheduler_->update(cost, batch);

    if(scheduler_->saving())
      this->save();

    if(scheduler_->validating()) {
      if(mvAvg_) {
        mvAvgGraph_->reuseWorkspace(graph_);
        scheduler_->validate({mvAvgGraph_});
      } else {
        scheduler_->validate({graph_});
      }
    }
  }
}
}
