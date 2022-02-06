#include "training/graph_group_singleton.h"

namespace marian {

void SingletonGraph::setScheduler(Ptr<Scheduler> scheduler) {
  scheduler_ = scheduler;
  // optimizer has to be registered last to see changes of learning rate
  scheduler_->registerTrainingObserver(scheduler_);
  for(auto opt : optimizerShards_)
    scheduler_->registerTrainingObserver(opt);
}

void SingletonGraph::execute(Ptr<data::Batch> batch) {
  auto graph = graphs_[0];
  auto model = models_[0];
  auto opt   = optimizerShards_[0];

  auto lossNode = model->build(graph, batch);
  if(costScalingFactor_ != 1.f) {
    // for fp16 training, it's ok to go out of scope, we do not use the scaled version for anything
    auto scaledLoss = lossNode->loss() * costScalingFactor_;
  }

  graph->forward();
  graph->backward();

  bool noNanOrInf = true;
  if(costScaling_) {
    // Are there NaNs in the gradient?
    bool hasNan = false, hasInf = false;
    IsNaN(graph->params()->grads(), graph->allocator(), hasNan, hasInf);
    noNanOrInf = !(hasNan || hasInf);

    if(!noNanOrInf) // there was a NaN, decrease cost-scaling
      GraphGroup::decreaseCostScaleFactor();
  }

  if(noNanOrInf) // skip update if NaN was seen @TODO: repeat instead with smaller factor?
    opt->update(graph->params()->vals(),
                graph->params()->grads(),
                batch->wordsTrg(),
                costScalingFactor_);

  if(scheduler_) {
    scheduler_->update(*lossNode, batch);

    if(scheduler_->validating()) {
      swapWithSmoothed();
      scheduler_->validate(graphs_);
      swapWithSmoothed();
    }

    if(scheduler_->saving())
      this->save();
  }

  if(noNanOrInf)
    GraphGroup::increaseCostScaleFactor();
}

}  // namespace marian

