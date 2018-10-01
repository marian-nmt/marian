#include "training/graph_group_sync.h"
#include "tensors/tensor_operators.h"

namespace marian {

SyncGraphGroup::SyncGraphGroup(Ptr<Config> config)
    : GraphGroup(config),
      ExponentialSmoothing{options_->get<float>("exponential-smoothing")},
      delay_{options_->get<size_t>("optimizer-delay")} { // @TODO: rename to something else; delay means delayed updated, not accumulation

  mpi_ = initMPI(/*multiThreaded=*/false); // when not running under MPI, this will be a fake object that represents a one-MPI-process setup

  devices_ = options_->getDevices(mpi_->myMPIRank(), mpi_->numMPIProcesses());
  for(auto device : devices_) {
    auto graph = New<ExpressionGraph>();
    graph->setDevice(device);
    graph->reserveWorkspaceMB(options_->get<size_t>("workspace"));
    graph->getBackend()->setClip(options_->get<float>("clip-gemm"));

    graphs_.push_back(graph);
    shardOpt_.push_back(Optimizer(options_));
    builders_.push_back(models::from_config(options_, models::usage::training));
  }

  // Note: We may well end up with only one MPI process or only one graph per worker.
  // This part of the code will not special-case any of this here.
  // Rather, it is assumed that the communicator knows to reduce unnecessary transfers to no-ops.
  comm_ = createCommunicator(graphs_, /*noNccl=*/options_->get<bool>("no-nccl", false), /*mpi=*/mpi_);
}

void SyncGraphGroup::setScheduler(Ptr<Scheduler> scheduler) {
  scheduler_ = scheduler;
  // optimizer has to be registered last to see changes of learning rate
  scheduler_->registerTrainingObserver(scheduler_);

  for(auto opt : shardOpt_)
    scheduler_->registerTrainingObserver(opt);
}

void SyncGraphGroup::initialize(const Ptr<data::Batch>& exampleBatch) {
  // Initialize 0th graph with random weights in one forward step
  THREAD_GUARD(builders_[0]->build(graphs_[0], exampleBatch);
               graphs_[0]->forward(););

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
}

void SyncGraphGroup::initializeAvg() {
  Ptr<ExpressionGraph> graphAvg; // CPU-side temp
  std::string name = options_->get<std::string>("model");
  if(boost::filesystem::exists(name + ".orig.npz")) {
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

      // temporary for the cost
      // @BUGBUG: must exist before first allocations; must do stuff in initialize()
      //auto costTmp = graph->param("special:costTmp", {}, inits::zeros);
      //if (t == 0)
      //    costTmp->grad()->set(0.0f);

      if(subBatch) {
        auto costNode = builders_[localDeviceIndex]->build(graph, subBatch);
        graph->forward();
        localDeviceCosts[localDeviceIndex] += costNode->scalar();
        graph->backward(/*zero=*/t == 0); // only reset gradients to 0 if t = 0
        //// record cost in a gradient
        //using namespace functional;
        //Element(_1 += _2, costTmp->grad(), costNode->val()); // stick it into a fake gradient that gets aggregated
        //// @TODO: Complete this. We still need to move it back from grad to val, which is tricky due to sharding.
      }
      else { // empty batch: execute do-nothing fw-bw step for proper inits and resets
        graph->forward();
        graph->backward(/*zero=*/t == 0);
      }
    };

    comm_->foreach(forwardBackward); // compute gradients in parallel on each device. Aggregate if delay_ > 1.
  }
  // At this point, each device on eacn MPI process has a gradient aggregated over a subset of the sub-batches.

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

    // fetch the aggregated cost
    // @TODO: We are sharded here. Need to copy costTmp->grad() to costTmp->val().
    //auto costTmp = graphs_[idx]->param("special:costTmp", {}, inits::zeros);
    //costTmp->val()->copyFrom(costTmp->grad());
  };

  comm_->scatterReduce(); // reduce gradients across all devices (globally) into shards
  comm_->foreach(update); // per-shard model-update
  comm_->allGather();     // distribute param value shards back

  // cost across all local devices
  // @TODO: We should report cost aggregated over all MPI processes.
  float cost = 0;
  for(auto& c : localDeviceCosts)
    cost += c;
  // extrapolate cost across MPI processes
  // @TODO: This is a crude estimate. Rather, we should aggregate cost across all GPUs correctly; cf. gradient trick described above.
  // @TODO: If this is too crude, we can also resurrect the code from f68433 to loop over the local batches,
  // and then determine a correction factor based on actual counts. They are very close though across MPI processes.
  cost *= mpi_->numMPIProcesses();

  // if cost is average-based, we need to turn the sum over devices into an average as well
  if(options_->get<std::string>("cost-type") != "ce-sum")
    cost /= numSubBatches;

  if(scheduler_) {
    // track and log cost
    scheduler_->update(cost, subBatches);

    // save intermediate model to file
    if(scheduler_->saving()) {
      save();
    }

    // process valid data set
    if(scheduler_->validating()) {
      if(mvAvg_) {
        comm_->swapParams(paramsAvg_);
      }

      // safe, because all graphs are idle during validation with sync sgd
      scheduler_->validate(graphs_);

      if(mvAvg_) {
        comm_->swapParams(paramsAvg_);
      }
    }
  }
}

void SyncGraphGroup::load() {
  if(!options_->get<bool>("no-reload")) {
    std::string name = options_->get<std::string>("model");

    if(boost::filesystem::exists(name)) {
      if(scheduler_)
        scheduler_->load(name);

      std::string nameGraph = name;
      if(mvAvg_ && boost::filesystem::exists(name + ".orig.npz"))
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
        [&](const std::vector<float>& data, const OptimizerBase::ScatterStateSetFunc& setFn) {
          comm_->scatterState(data, setFn);
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

void SyncGraphGroup::save(bool final) {
  // do final validation
  if(final && scheduler_) {
    // bring the smoothed model in
    // Note that it is sharded. For multi-node, it is sharded over multiple machines, so this is a network access.
    if(mvAvg_ && paramsAvg_.size() > 0)
      comm_->swapParams(paramsAvg_);

    mpi_->barrier();
    if (mpi_->myMPIRank() == 0) { // in multi-node, only first MPI process saves the model (they are all identical)
      scheduler_->validate(graphs_, true);
    }
    mpi_->barrier();

    if(mvAvg_ && paramsAvg_.size() > 0)
      comm_->swapParams(paramsAvg_);
  }

  std::string name = options_->get<std::string>("model");

  // save original (unsmoothed) parameters as well
  // @TODO: Check whether we are reloading the correct file (the unsmoothed one).
  if(mvAvg_ && paramsAvg_.size() > 0) {
    // Save the original parameters in model.npz.orig.npz
    if (mpi_->myMPIRank() == 0) // only save from one MPI process
      builders_[0]->save(graphs_[0], name + ".orig.npz", true);
    // Switch to the averaged parameters
    comm_->swapParams(paramsAvg_); // note: the smoothed model is sharded across GPUs, and across MPI processes if applicablenode. This brings it into MPI process[0].device[0]
  }

  // save main model file
  // @TODO: do we need a barrier here as wel?
  if (mpi_->myMPIRank() == 0) { // only save from one MPI process
    // if not overwrite then save a copy with number of updates in the model pathname
    if(!options_->get<bool>("overwrite") && !final) {
      std::string numberOfBatches
          = scheduler_ ? std::to_string(scheduler_->numberOfBatches())
                       : "unknown";
      std::string nameOverwrite = name;
      nameOverwrite.replace(
          name.size() - 4, 4, ".iter" + numberOfBatches + ".npz");
      builders_[0]->save(graphs_[0], nameOverwrite);
    }
    // save main model file
    builders_[0]->save(graphs_[0], name, true);
    if (scheduler_)
      scheduler_->save(name);
  }

  if(mvAvg_ && paramsAvg_.size() > 0)
    // Switch back to the original parameters
    comm_->swapParams(paramsAvg_);

  shardOpt_[0]->save(name + ".optimizer.npz", shardOpt_,
    [&](const OptimizerBase::GatherStateGetFunc& getFn) {
      return comm_->gatherState(getFn);
    });
}

}  // namespace marian
