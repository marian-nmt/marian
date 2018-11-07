#include "training/graph_group_sync.h"

namespace marian {

SyncGraphGroup::SyncGraphGroup(Ptr<Options> config)
    : GraphGroup(config),
      ExponentialSmoothing{options_->get<float>("exponential-smoothing")},
      delay_{options_->get<size_t>("optimizer-delay")} { // @TODO: rename to something else; delay means delayed updated, not accumulation

  mpi_ = initMPI(/*multiThreaded=*/false); // when not running under MPI, this will be a fake object that represents a one-MPI-process setup

  devices_ = Config::getDevices(options_, mpi_->myMPIRank(), mpi_->numMPIProcesses());
  for(auto device : devices_) {
    auto graph = New<ExpressionGraph>();
    graph->setDevice(device);
    graph->reserveWorkspaceMB(options_->get<size_t>("workspace"));
    graph->getBackend()->setClip(options_->get<float>("clip-gemm"));

    graphs_.push_back(graph);
    shardOpt_.push_back(Optimizer(options_));
    builders_.push_back(models::from_options(options_, models::usage::training));
  }

  // Note: We may well end up with only one MPI process or only one graph per worker.
  // This part of the code will not special-case any of this here.
  // Rather, it is assumed that the communicator knows to reduce unnecessary transfers to no-ops.
  comm_ = createCommunicator(graphs_, /*noNccl=*/options_->get<bool>("no-nccl", false), /*mpi=*/mpi_);
}

void SyncGraphGroup::setScheduler(Ptr<Scheduler> scheduler) /*override*/ {
  scheduler_ = scheduler;
  // optimizer has to be registered last to see changes of learning rate
  // @TODO: ^^Fix this comment. Either it refers to the scheduler, or it should be moved. Which one?
  scheduler_->registerTrainingObserver(scheduler_);

  for(auto opt : shardOpt_)
    scheduler_->registerTrainingObserver(opt);
}

void SyncGraphGroup::initialize(const Ptr<data::Batch>& exampleBatch) {
  // Initialize 0th graph with random weights in one forward step
  // @TODO: Why do we need the THREAD_GUARD here? Why not run this on the main thread?
  THREAD_GUARD({
    builders_[0]->build(graphs_[0], exampleBatch);
    graphs_[0]->forward();
  });

  // Copy weights from 0th graph to all other graphs
  // to have equal weights across devices
  ThreadPool pool(graphs_.size() - 1, graphs_.size() - 1);
  for(size_t i = 1; i < graphs_.size(); ++i) {
    auto init = [&](size_t i) {
      // initialize t-th graph and weights
      builders_[i]->build(graphs_[i], exampleBatch);
      graphs_[i]->forward();
      // overwrite weights of t-th graph with weights from 0th graph
      graphs_[i]->params()->vals()->copyFrom(graphs_[0]->params()->vals());
    };
    pool.enqueue(init, i);
  }
  // ThreadPool destructor waits until completion of all tasks.
  // @TODO: can we use comm_->foreach()?
}

void SyncGraphGroup::initializeAvg() {
  Ptr<ExpressionGraph> graphAvg; // CPU-side temp
  std::string name = options_->get<std::string>("model");
  if(filesystem::exists(name + ".orig.npz")) {
    // Load the averaged parameters into a temporary graph
    graphAvg = New<ExpressionGraph>();
    graphAvg->setDevice({0, DeviceType::cpu});
    graphAvg->load(name, false);
    graphAvg->forward(); // initialize parameters if needed
  }

  auto init = [&](size_t localDeviceIndex, size_t begin, size_t end) {
    size_t size = end-begin;

    // get the device-specific allocator
    auto paramsAllocator = New<TensorAllocator>(graphs_[localDeviceIndex]->getBackend());
    paramsAllocs_[localDeviceIndex] = paramsAllocator;

    paramsAllocator->reserveExact(size * sizeof(float));

    Tensor paramAvg;
    paramsAllocator->allocate(paramAvg, {1, (int)size});
    paramsAvg_[localDeviceIndex] = paramAvg;

    if(graphAvg)
      paramAvg->copyFrom(graphAvg  ->params()->vals()->subtensor(begin, size));
    else
      paramAvg->copyFrom(graphs_[0]->params()->vals()->subtensor(begin, size));

    // note: for multi-node, graphAvg and graphs_[0] contain a complete copy, from which
    // each MPI process copies only part into its respective shard(s)
  };

  paramsAllocs_.resize(graphs_.size()); // allocators
  paramsAvg_.resize(graphs_.size());    // averaged parameters (shards; distributed over MPI processes if applicable)
  comm_->foreach(init, /*parallel=*/false); // @TODO: is sequential operation necessary here? (is the allocation stuff sufficiently reentrant or thread-separated?)
}

Ptr<data::BatchStats> SyncGraphGroup::collectStats() {
  // @TODO: This should only run on MPI process 0. Also we can share vv this vv expression with update().
  size_t multiplier = devices_.size() * mpi_->numMPIProcesses() * delay_;
  return GraphGroup::collectStats(graphs_[0], builders_[0], multiplier);
}

void SyncGraphGroup::update(Ptr<data::Batch> batch) /*override*/ {
  ABORT_IF(finalized_, "Training has already finished.");

  // distribute the batch over (delay, local device, MPI rank)
  size_t numSubBatches = delay_ * devices_.size() * mpi_->numMPIProcesses();
  auto subBatches = batch->split(numSubBatches);
  subBatches.resize(numSubBatches); // pad with nullptrs if out of data

  // Helper to access the subBatches array
  auto getSubBatch = [&](size_t t, size_t localDeviceIndex, size_t rank) {
    // 't' (the delay) should be slowest changing dimension. If subBatches are sorted by
    // length, then grouping sentences of similar length into the same delay step can
    // reduce unnecessary time spent in padding.
    return subBatches[(t * mpi_->numMPIProcesses() + rank) * devices_.size() + localDeviceIndex];
  };

  // Upon very first execution, reset everything
  if(first_) {
    LOG(debug, "[{}] Processing first minibatch. Batches are processed as {} processes x {} GPUs/process x {} delay steps.",
         mpi_->idStr(), mpi_->numMPIProcesses(), devices_.size(), delay_);
    initialize(subBatches.front());
    if(mvAvg_ && paramsAvg_.empty())
      initializeAvg();
    first_ = false;
  }

  // Compute gradients
  // This happens in multiple steps in case of delay_ > 1.
  std::vector<float> localDeviceCosts(devices_.size(), 0.f); // [local device index] aggregate cost for each local device
  for (size_t t = 0; t < delay_; t++) {
    // Execute single forward/backward step
    auto forwardBackward = [&](size_t localDeviceIndex, size_t /*begin*/, size_t /*end*/) {
      auto graph = graphs_[localDeviceIndex];
      auto subBatch = getSubBatch(t, localDeviceIndex, mpi_->myMPIRank());

      if(subBatch) {
        timer::Timer timer;
        auto costNode = builders_[localDeviceIndex]->build(graph, subBatch);
        //LOG(info, timer.format(2, "after build: %ws"));
        graph->forward();
        //LOG(info, timer.format(2, "after forward (no sync): %ws"));
        localDeviceCosts[localDeviceIndex] += costNode->scalar();
        graph->backward(/*zero=*/t == 0); // only reset gradients to 0 if t = 0
        //LOG(info, timer.format(2, "after backward (no sync): %ws"));
        //localDeviceCosts[localDeviceIndex] += costNode->scalar(); // moved here for time measurements; @TODO: move this back
        //LOG(info, timer.format(2, "after scalar() (that's a sync): %ws"));
      }
      else { // empty batch: execute do-nothing fw-bw step for proper inits and resets
        graph->forward();
        graph->backward(/*zero=*/t == 0);
      }
    };

    comm_->foreach(forwardBackward); // compute gradients in parallel on each device. Aggregate if delay_ > 1.
  }
  // At this point, each device on each MPI process has a gradient aggregated over a subset of the sub-batches.

  // Update parameter shard with gradient shard
  auto update = [&](size_t idx, size_t begin, size_t end) {
    auto curGrad = graphs_[idx]->params()->grads()->subtensor(begin, end-begin);
    auto curParam = graphs_[idx]->params()->vals()->subtensor(begin, end-begin);

    // if individual gradients were averages, then need to average again over all subBatches
    auto div = subBatches.size();
    if (options_->get<std::string>("cost-type") == "ce-sum")
      div = 1;
    if(div != 1) {
      using namespace functional;
      Element(_1 = _1 / (float)div, curGrad);
    }

    // actual model update
    shardOpt_[idx]->update(curParam, curGrad);

    if(mvAvg_)
      updateAvgParams(
          paramsAvg_[idx], curParam, scheduler_->numberOfBatches());
  };

  timer::Timer timer;
  comm_->scatterReduce(); // reduce gradients across all devices (globally) into shards
  //LOG(info, timer.format(2, "after scatterReduce (has sync): %ws"));
  comm_->foreach(update); // per-shard model-update
  //LOG(info, timer.format(2, "after model update (no sync): %ws"));
  //graphs_.front()->getBackend()->synchronize(); // @TODO: This is strictly for time measurement. Make sure it doesn't accidentally stay in here!!
  //LOG(info, timer.format(2, "after model update sync (which is unnecessary except for time measurements): %ws"));
  comm_->allGather();     // distribute param value shards back
  //LOG(info, timer.format(2, "after allGather (has sync): %ws"));

  // cost across all local devices (scheduler will aggregate cross-process)
  float localCost = 0;
  for(auto& c : localDeviceCosts) // localDeviceCosts is already summed up over delay steps
    localCost += c;

  // if localCost is average-based, we need to turn the sum over devices into an average as well
  if(options_->get<std::string>("cost-type") != "ce-sum")
    localCost /= numSubBatches;

  if(scheduler_) {
    // track and log localCost
    scheduler_->update(localCost, subBatches, mpi_);

    // save intermediate model (and optimizer state) to file
    if(scheduler_->saving())
      save();

    // process valid data set
    // This may save a model as well.
    if(scheduler_->validating()) {
      swapParamsAvg();
      if (isMainProcess())
        scheduler_->validate(graphs_);
      swapParamsAvg();
    }
  }
}

void SyncGraphGroup::load() /*override*/ {

  // This function loads the main parameters in the graphs.
  // In case of exponential smoothing, we also need to restore paramsAvg_.
  // That is done lazily inside initializeAvg(), see there.

  if(!options_->get<bool>("no-reload")) {
    std::string name = options_->get<std::string>("model");

    if(filesystem::exists(name)) {
      if(scheduler_)
        scheduler_->load(name);

      std::string nameGraph = name;
      if(mvAvg_ && filesystem::exists(name + ".orig.npz"))
        // Load the original parameters from model.npz.orig.npz
        nameGraph += ".orig.npz";

      size_t i = 0;
      for(auto graph : graphs_)
        builders_[i++]->load(graph, nameGraph); // we just load it N times from disk (it'll be in disk cache after the first)

      // @TODO: probably we want to have the list of DeviceIds as an attribute
      std::vector<Ptr<Backend>> backends;
      for(auto graph : graphs_)
        backends.push_back(graph->getBackend());
      shardOpt_[0]->load(name + ".optimizer.npz", shardOpt_, backends,
        [&](const std::vector<float>& optimizerStateVector, const OptimizerBase::ScatterStateSetFunc& setShardFn) {
          comm_->scatterState(optimizerStateVector, setShardFn);
        });
    } else if(options_->has("pretrained-model")) {
      std::string nameInit = options_->get<std::string>("pretrained-model");
      LOG(info,
          "Initialize model weights with the pre-trained model {}",
          nameInit);

      size_t i = 0;
      for(auto graph : graphs_)
        builders_[i++]->load(graph, nameInit, false);
    }
  }
}

void SyncGraphGroup::save(bool final) /*override*/ {
  barrier(); // (for better grouping of log messages)
  //LOG(info, "[{}] save() line {}!", this->mpi_->idStr(), __LINE__);
  // do final validation
  if(final && scheduler_) {
    // bring the smoothed model in
    // Note that it is sharded. For multi-node, it is sharded over multiple machines, so this is a network access.
    // Also note that the swap must run on all MPI processes concurrently, although only one actually validates.
    //LOG(info, "[{}] save() line {}", this->mpi_->idStr(), __LINE__);
    swapParamsAvg();
    //LOG(info, "[{}] save() line {}", this->mpi_->idStr(), __LINE__);
    if (isMainProcess()) // in multi-node, only first MPI process saves the model (they are all identical)
      scheduler_->validate(graphs_, true);
    //LOG(info, "[{}] save() line {}", this->mpi_->idStr(), __LINE__);
    swapParamsAvg();
    //LOG(info, "[{}] save() line {}", this->mpi_->idStr(), __LINE__);
  }

  std::string name = options_->get<std::string>("model");

  //LOG(info, "[{}] save() line {}", this->mpi_->idStr(), __LINE__);
  barrier(); // (for better grouping of log messages)
  //LOG(info, "[{}] save() line {}", this->mpi_->idStr(), __LINE__);
  // if smoothing then save original (unsmoothed) parameters as well
  // @TODO: Check whether we are reloading the correct file (the unsmoothed one).
  if(mvAvg_ && paramsAvg_.size() > 0 && isMainProcess()) // only save from one MPI process
    // Save the original parameters in model.npz.orig.npz
    builders_[0]->save(graphs_[0], name + ".orig.npz", true);

  // Temporarily switch to the averaged parameters
  // Note: the smoothed model is sharded across GPUs, and across MPI processes if applicable. This brings it into MPI process[*].device[*]
  //LOG(info, "[{}] save() line {}", this->mpi_->idStr(), __LINE__);
  swapParamsAvg();
  //LOG(info, "[{}] save() line {}", this->mpi_->idStr(), __LINE__);

  // save main model file
  if (isMainProcess()) { // only save from one MPI process
    // if not overwrite then save a copy with number of updates in the model pathname
    //LOG(info, "[{}] save() line {}", this->mpi_->idStr(), __LINE__);
    if(!options_->get<bool>("overwrite") && !final) {
      std::string numberOfBatches
          = scheduler_ ? std::to_string(scheduler_->numberOfBatches())
                       : "unknown";
      std::string nameOverwrite = name;
      nameOverwrite.replace(name.size() - 4, 4, ".iter" + numberOfBatches + ".npz"); // @TODO: use insert?
      builders_[0]->save(graphs_[0], nameOverwrite);
    }
    //LOG(info, "[{}] save() line {}", this->mpi_->idStr(), __LINE__);
    // save main model file
    builders_[0]->save(graphs_[0], name, true);
    //LOG(info, "[{}] save() line {}", this->mpi_->idStr(), __LINE__);
    // save scheduler-related state
    if (scheduler_)
      scheduler_->save(name);
    //LOG(info, "[{}] save() line {}", this->mpi_->idStr(), __LINE__);
  }

  // Switch back to the original parameters
  //LOG(info, "[{}] save() line {}", this->mpi_->idStr(), __LINE__);
  swapParamsAvg();
  //LOG(info, "[{}] save() line {}", this->mpi_->idStr(), __LINE__);

#if 0 // temporary, for testing of saving distributed models; must be identical to .orig.npz
  if(mvAvg_ && paramsAvg_.size() > 0 && isMainProcess())
    builders_[0]->save(graphs_[0], name + ".orig_after_swapping.npz", true);
#endif
  //LOG(info, "[{}] save() line {}", this->mpi_->idStr(), __LINE__);
  barrier(); // (for better grouping of log messages)
  //LOG(info, "[{}] save() line {}", this->mpi_->idStr(), __LINE__);

  // persist optimizer state
  //LOG(info, "[{}] save() line {}", this->mpi_->idStr(), __LINE__);
  shardOpt_[0]->save(name + ".optimizer.npz", shardOpt_,
    [&](const OptimizerBase::GatherStateGetFunc& getShardFn) {
      return comm_->gatherState(getShardFn);
    },
    isMainProcess());
  //LOG(info, "[{}] save() line {}", this->mpi_->idStr(), __LINE__);

  barrier(); // (for better grouping of log messages)
  //LOG(info, "[{}] save() line {}", this->mpi_->idStr(), __LINE__);
}

}  // namespace marian
