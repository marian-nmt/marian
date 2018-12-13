#include "training/graph_group_sync.h"

namespace marian {

SyncGraphGroup::SyncGraphGroup(Ptr<Options> config)
    : GraphGroup(config),
      ExponentialSmoothing(config),
      delay_{options_->get<double>("optimizer-delay")} { // @TODO: rename to something else; delay means delayed updated, not accumulation

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

  auto type = utils::toUpper(devices_.front().typeAsString()) + "s";
  if (mpi_->numMPIProcesses() > 1)
    LOG(info, "[training] Using {} {}, distributed over {} MPI processes", mpi_->numMPIProcesses() * devices_.size(), type, mpi_->numMPIProcesses());
  else
    LOG(info, "[training] Using {} {}", devices_.size(), type);
}

void SyncGraphGroup::setScheduler(Ptr<Scheduler> scheduler) /*override*/ {
  validate();
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
  // @TODO: This is an incompatible change. Decide how to handle that.
  //size_t multiplier = devices_.size() * mpi_->numMPIProcesses() * delay_;
  return GraphGroup::collectStats(graphs_[0], builders_[0]/*, multiplier*/);
}

// helper for MB scaling: quantize the ratio with a given error margin
static double roundUpRatio(double ratio) {
  if (ratio == 0)
    return ratio;
  // find largest power of two that fits into ratio
  double p = 1;
  while (p*2 < ratio)
    p *= 2;
  // round up to nearest multiple of a largest power of 2 where relative error is within margin
  // 25% error margin seems acceptable:
  //  - using a 25% larger MB size should not break convergence
  //  - @TODO: not using the first 25% of the next block is OK since those are dominated by data exchange
  double maxError = 0.25;
  while (p >= 1) {
    double proposedRatio = ceil(ratio / p) * p;
    double error = (proposedRatio - ratio) / ratio;
    if (fabs(error) <= maxError)
      return proposedRatio;
    p /= 2;
  }
  return ratio;
}

// helper routine that handles accumulation and load-balancing of sub-batches to fill all devices
// It adds 'newBatch' to 'pendingBatches_', and if sufficient batches have been queued, then
// returns 'pendingBatches_' in 'subBatches' and resets it. If not, it returns false.
bool SyncGraphGroup::tryGetSubBatches(Ptr<data::Batch> newBatch, std::vector<Ptr<data::Batch>>& subBatches) {
  pendingBatches_.push_back(newBatch);
  size_t warpSize = devices_.size() * mpi_->numMPIProcesses(); // warp := set of batches processed concurrently across GPus and workers

  size_t pendingTrgWords = 0; // diagnosics only: compute how many target labels are pending so far
  for (const auto& batch : pendingBatches_)
    pendingTrgWords += batch->wordsTrg();

  // MB-size warm-up and dynamic scaling
  double ratio;
  bool isDynamic = scheduler_->tryGetDynamicMBSizeMultiplier(ratio);
  if (isDynamic)
    ratio = roundUpRatio(ratio); // round up to full batches if within a certain error margin  --@BUGBUG: Not invariant w.r.t. GPU size, as ratio is relative to what fits into 1 GPU
  else   // if dynamic scaling not enabled, then fill each GPU with a batch
    ratio = delay_ * (double)warpSize; // note: delay_ may be fractional

  // adjust for reference batch size if given
  // At progress == mbWarmup.n (ratio=1), we would like to have refBatchLabels instead of whichever
  // the actual batch size is. We approximate the latter as typicalTrgBatchWords, and scale ratio accordingly.
  auto refBatchLabels = options_->get<size_t>("mini-batch-words-ref");
  if (refBatchLabels != 0) {
    auto typicalTrgBatchWords = scheduler_->getTypicalTrgBatchWords();
    LOG_ONCE(info, "[scheduler] Scaling to {} reference labels. Typical actual batch words is {}", refBatchLabels, typicalTrgBatchWords);
    ABORT_IF(typicalTrgBatchWords == 0, "dynamic scaling with words target requires MB size to be known in words"); // happens if MB size is specified in sentences
    ratio *= (double)refBatchLabels / (double)typicalTrgBatchWords;
  }

  if (pendingBatches_.size() < ratio)
    return false; // not enough data yet

  // now we have enough to fill at least 'ratio' batches
  if (pendingBatches_.size() == ratio)
    return true; // nothing to do, e.g. warm-up not enabled

  // warm-up is happening
  LOG_ONCE(info, "[training] Mini-batch-warmup enabled");

  // shorten all batches a little to accurately reflect ratio
  // e.g. ratio = 3.3 for 4 batches: Reduce each by 3.3/4
  // Alternatively, we could just shorten the last 'warp', but that would not be invariant to warp size.
  size_t before = 0, after = 0;
  for (auto& batch : pendingBatches_) {
    auto reducedBatchSize = (size_t)ceil((double)batch->size() * ratio / (double)pendingBatches_.size());
    size_t minSize = 1;
    if (pendingBatches_.size() == 1) { // enforce a minimum (only needed/correct if still in first batch)
      size_t minTrgWords = 256;    // don't go below this number of target words, as it seems excessive  --@TODO: parameterize?
      minSize = 1 + (minTrgWords * batch->size() - 1) / batch->wordsTrg(); // approximately convert minTrgWords into a #sentences
    }
    reducedBatchSize = std::max(reducedBatchSize, minSize);
    before += batch->wordsTrg();
    if (reducedBatchSize < batch->size())
      batch = batch->split(/*numSubBatches=*/1, reducedBatchSize).front();
    after += batch->wordsTrg();
  }

  // load-balance: distribute the last numWarps-group's batches over GPUs
  // This is tricky since batches do not have the same length, therefore we can only split, but not merge.
  auto numWarps = (pendingBatches_.size() - 1) / warpSize + 1; // = ceil(#buffers / (#GPUs * #workers))
  auto availableBatches = numWarps * warpSize; // we got this many GPUs anyways, so we better make use of them
  if (pendingBatches_.size() < availableBatches) {
    // we are not using all available GPUs -> try to load-balance a bit better
    auto fullBatches = (numWarps - 1) * warpSize;
    auto expandLast = pendingBatches_.size() - fullBatches;
    auto toLast = availableBatches - fullBatches;
    LOG(info, "attempt to redistribute {} last batches over {}", expandLast, toLast);
    auto splitInto = toLast / expandLast; // unfortunately we can only split in integer ratios
    // @TODO: We can do better since the last batch is typically smaller.
    if (splitInto > 1) {
      // split each of last numWarps's batches into 'splitInto' batches
      // pop them first
      std::vector<Ptr<data::Batch>> batchesToSplit;
      while (pendingBatches_.size() > fullBatches) {
        batchesToSplit.push_back(pendingBatches_.back());
        pendingBatches_.pop_back();
      }
      // now split them
      for (auto& batchToSplit : batchesToSplit) {
        LOG(info, "{}-way splitting batchToSplit with size {}", splitInto, batchToSplit->size());
        auto splitBatches = batchToSplit->split(splitInto);
        for (auto& splitBatch : splitBatches) {
          LOG(info, " -> getting batchToSplit with size {}", splitBatch->size());
          pendingBatches_.push_back(splitBatch);
        }
      }
    }
    ABORT_IF(pendingBatches_.size() > availableBatches, "somehow split into too many batches??");
  }
  subBatches = std::move(pendingBatches_);

  // @TODO: sort by width, so that in case of delay > 1, each GPU gets about the same size
  return true;
}

void SyncGraphGroup::update(Ptr<data::Batch> newBatch) /*override*/ {
  validate();

  std::vector<Ptr<data::Batch>> subBatches;
  bool gotSubBatches = tryGetSubBatches(newBatch, subBatches);

  // not enough data yet: return right away
  if (!gotSubBatches)
    return;

  // Helper to access the subBatches array
  auto getSubBatch = [&](size_t t, size_t localDeviceIndex, size_t rank) -> Ptr<data::Batch> {
    // 't' (the delay) should be slowest changing dimension. If subBatches are sorted by
    // length, then grouping sentences of similar length into the same delay step can
    // reduce unnecessary time spent in padding.
    auto index = (t * mpi_->numMPIProcesses() + rank) * devices_.size() + localDeviceIndex;
    if (index < subBatches.size())
      return subBatches[index];
    else
      return nullptr;
  };

  // Upon very first execution, reset everything
  if(first_) {
    LOG(info, "[training] Processing first minibatch. Batches are processed as {} processes x {} GPUs/process",
        mpi_->numMPIProcesses(), devices_.size());
    initialize(subBatches.front());
    if(mvAvg_ && paramsAvg_.empty())
      initializeAvg();
    first_ = false;
  }

  // Compute gradients
  // This happens in multiple steps in case of delay > 1.
  std::vector<float> localDeviceCosts(devices_.size(), 0.f); // [local device index] aggregate cost for each local device
  for (size_t t = 0; getSubBatch(t, 0, 0); t++) { // @TODO: rename 't' to 'delay'
    // Execute single forward/backward step
    auto forwardBackward = [&](size_t localDeviceIndex, size_t /*begin*/, size_t /*end*/) {
      auto graph = graphs_[localDeviceIndex];
      auto subBatch = getSubBatch(t, localDeviceIndex, mpi_->myMPIRank());

      if(subBatch) {
        auto costNode = builders_[localDeviceIndex]->build(graph, subBatch);
        graph->forward();
        localDeviceCosts[localDeviceIndex] += costNode->scalar();
        graph->backward(/*zero=*/t == 0); // only reset gradients to 0 if t = 0
      }
      else { // empty batch: execute do-nothing fw-bw step for proper inits and resets
#if 1   // @TODO: double-check whether the #else branch is the same; and if so, use it instead
        graph->params()->allocateBackward();
        if (t == 0) // these have already been sized
          graph->params()->set_zero_adjoint();
#else
        graph->clear(); // instead of build()
        graph->forward();
        graph->backward(/*zero=*/t == 0);
#endif
      }
    };

    comm_->foreach(forwardBackward); // compute gradients in parallel on each device. Aggregate if delay > 1.
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

    // determine num words for dynamic hyper-parameter adjustment
    size_t mbWords = OptimizerBase::mbSizeNotProvided;
    if (options_->get<std::string>("cost-type") == "ce-sum") { // presently only supported for ce-sum
      mbWords = 0;
      for (const auto& batch : subBatches)
        mbWords += batch->words(-1);  // @TODO: use wordsTrg (it's the same)
    }

    // actual model update
    shardOpt_[idx]->update(curParam, curGrad, mbWords);

    if(mvAvg_)
      updateAvgParams(
          paramsAvg_[idx], curParam, scheduler_->numberOfBatches(), mbWords);
  };

  comm_->scatterReduce(); // reduce gradients across all devices (globally) into shards
  comm_->foreach(update); // per-shard model-update
  comm_->allGather();     // distribute param value shards back

  // cost across all local devices (scheduler will aggregate cross-process)
  float localCost = 0;
  for(auto& c : localDeviceCosts) // localDeviceCosts is already summed up over delay steps
    localCost += c;

  // if localCost is average-based, we need to turn the sum over devices into an average as well
  if(options_->get<std::string>("cost-type") != "ce-sum")
    localCost /= subBatches.size();

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
  validate();

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
      LOG(info, "[training] Model reloaded from {}", name);
    } else if(options_->has("pretrained-model")) {
      std::string nameInit = options_->get<std::string>("pretrained-model");
      LOG(info,
          "[training] Initializing model weights with the pre-trained model {}",
          nameInit);

      size_t i = 0;
      for(auto graph : graphs_)
        builders_[i++]->load(graph, nameInit, false);
    }
  }
}

void SyncGraphGroup::save(bool final) /*override*/ {
  validate();
  barrier(); // (for better grouping of log messages)
  // do final validation
  if(final && scheduler_) {
    // bring the smoothed model in
    // Note that it is sharded. For multi-node, it is sharded over multiple machines, so this is a network access.
    // Also note that the swap must run on all MPI processes concurrently, although only one actually validates.
    swapParamsAvg();
    if (isMainProcess()) // in multi-node, only first MPI process saves the model (they are all identical)
      scheduler_->validate(graphs_, true);
    swapParamsAvg();
  }

  std::string name = options_->get<std::string>("model");

  barrier(); // (for better grouping of log messages)
  // if smoothing then save original (unsmoothed) parameters as well
  if(mvAvg_ && paramsAvg_.size() > 0 && isMainProcess()) // only save from one MPI process
    // Save the original parameters in model.npz.orig.npz
    builders_[0]->save(graphs_[0], name + ".orig.npz", true);

  // Temporarily switch to the averaged parameters
  // Note: the smoothed model is sharded across GPUs, and across MPI processes if applicable. This brings it into MPI process[*].device[*]
  swapParamsAvg();

  // save main model file
  if (isMainProcess()) { // only save from one MPI process
    // if not overwrite then save a copy with number of updates in the model pathname
    if(!options_->get<bool>("overwrite") && !final) {
      std::string numberOfBatches
          = scheduler_ ? std::to_string(scheduler_->numberOfBatches())
                       : "unknown";
      std::string nameOverwrite = name;
      nameOverwrite.replace(name.size() - 4, 4, ".iter" + numberOfBatches + ".npz"); // @TODO: use insert?
      builders_[0]->save(graphs_[0], nameOverwrite);
    }
    // save main model file
    builders_[0]->save(graphs_[0], name, true);
    // save scheduler-related state
    if (scheduler_)
      scheduler_->save(name);
  }

  // Switch back to the original parameters
  swapParamsAvg();

  barrier(); // (for better grouping of log messages)

  // persist optimizer state
  LOG(info, "[{}] save() line {}", this->mpi_->idStr(), __LINE__);
  shardOpt_[0]->save(name + ".optimizer.npz", shardOpt_,
    [&](const OptimizerBase::GatherStateGetFunc& getShardFn) {
      return comm_->gatherState(getShardFn);
    },
    isMainProcess());
  LOG(info, "[{}] save() line {}", this->mpi_->idStr(), __LINE__);

  barrier(); // (for better grouping of log messages)
}

void SyncGraphGroup::finalize() /*override*/ {
  validate();
  finalizeMPI(std::move(mpi_));
  Base::finalize();
}

}  // namespace marian
