#include "training/graph_group.h"

namespace marian {

GraphGroup::GraphGroup(Ptr<Options> options, Ptr<IMPIWrapper> mpi)
  : options_(options),
    mpi_(mpi),
    devices_(Config::getDevices(options, mpi->myMPIRank(), mpi->numMPIProcesses())),
    shardingMode_(getShardingMode(options_, mpi)),
    mbRoundUp_(options_->get<bool>("mini-batch-round-up", true)) {
  if(options_->hasAndNotEmpty("cost-scaling")) {
    auto vcs = options_->get<std::vector<std::string>>("cost-scaling");

    costScaling_                                 = true;
    costScalingFactor_                           = std::stof( vcs[0]);
    if(vcs.size() > 1) costScalingFreq_          = std::stoul(vcs[1]);
    if(vcs.size() > 2) costScalingMultiplier_    = std::stof( vcs[2]);
    if(vcs.size() > 3) costScalingFactorMinimum_ = std::stof( vcs[3]);
    
    LOG_ONCE(info,
             "Training with cost scaling - factor: {}, frequency: {}, multiplier: {}, minimum: {}",
             costScalingFactor_,
             costScalingFreq_,
             costScalingMultiplier_,
             costScalingFactorMinimum_);
  }

  if(options_->hasAndNotEmpty("dynamic-gradient-scaling")) {
    auto vgc = options_->get<std::vector<std::string>>("dynamic-gradient-scaling");
    dynamicGradientScaling_ = true;

    if(vgc.size() > 0) dynamicGradientScalingFactor_  = std::stof(vgc[0]);
    if(vgc.size() > 1) dynamicGradientScalingUseLogs_ = vgc[1] == "log";
    if(vgc.size() > 2) dynamicGradientScalingFadeout_ = std::stoul(vgc[2]);

    LOG_ONCE(info,
             "Re-scaling gradient to have average gradient norm if (log={}) gradient norm diverges from average by {} sigmas",
             dynamicGradientScalingUseLogs_,
             dynamicGradientScalingFactor_);
    if(dynamicGradientScalingFadeout_ > 0)
      LOG_ONCE(info,
               "Dynamic gradient re-scaling will fade out linearly after {} updates",
               dynamicGradientScalingFadeout_);
  }

  if(options_->get<bool>("check-gradient-nan")) {
    checkGradientNan_ = true;
    LOG_ONCE(info, "Checking gradient for NaN");
  }

  initGraphsAndOpts();

  // Note: We may well end up with only one MPI process or only one graph per worker.
  // This part of the code will not special-case any of this here.
  // Rather, it is assumed that the communicator knows to reduce unnecessary transfers to no-ops.
  // @TODO: createCommunicator(options, ...)
  comm_ = createCommunicator(graphs_,
                             /*noNccl=*/options_->get<bool>("no-nccl", false),
                             shardingMode_,
                             /*mpi=*/mpi_);

  auto formattedDeviceType = utils::utf8ToUpper(devices_.front().typeAsString()) + "s";
  if (mpi_->numMPIProcesses() > 1)
    LOG(info, "[training] Using {} {}, distributed over {} MPI processes", mpi_->numMPIProcesses() * devices_.size(), formattedDeviceType, mpi_->numMPIProcesses());
  else
    LOG(info, "[training] Using {} {}", devices_.size(), formattedDeviceType);
}

void GraphGroup::initGraphsAndOpts() {
  for(auto device : devices_) {
    auto graph = New<ExpressionGraph>();
    
    // @TODO: validate precisions in config
    auto precisions = options_->get<std::vector<std::string>>("precision");
    Type parameterType = typeFromString(precisions[0]);

    graph->setDefaultElementType(parameterType);
    graph->setCheckpointing(options_->get<bool>("gradient-checkpointing"));

    if(options_->get<bool>("check-nan")) // @TODO: add to other places
      graph->setThrowNaN(true);

    graph->setDevice(device);
    
    graph->reserveWorkspaceMB(options_->get<size_t>("workspace"));

    graphs_.push_back(graph);

    optimizerShards_.push_back(Optimizer(options_));
    models_.push_back(models::createCriterionFunctionFromOptions(options_, models::usage::training));
  }
}

// increase cost-scaling factor if no NaN has been detected for a
// given number of iterations. Usually we increase by 2 which adds
// one more bit for precision.
void GraphGroup::increaseCostScaleFactor() {
  if(!costScaling_)
    return;

  noNanSeen_++;

  size_t total = nanSeen_ + noNanSeen_;

  if(noNanSeen_ % costScalingFreq_ == 0) {
    costScalingFactor_ *= costScalingMultiplier_;
    if(isMainProcess())
      LOG(debug, "No NaN/Inf after {} gradient updates. Increasing cost-scaling factor to {}", total, costScalingFactor_);

    // Resetting counts after cost-scale change
    noNanSeen_ = 0;
    nanSeen_ = 0;
  }
}

// call when a NaN was seen to decrease cost-scaling factor
void GraphGroup::decreaseCostScaleFactor() {
  if(!costScaling_)
    return;

  nanSeen_++;
  
  size_t total = nanSeen_ + noNanSeen_;

  // do not reduce cost-scaling factor below minimum
  if(costScalingFactor_ > costScalingFactorMinimum_)
    costScalingFactor_ /= costScalingMultiplier_;

  if(isMainProcess()) {
    if(costScalingFactor_ > costScalingFactorMinimum_)
      LOG(debug, "Seen NaN/Inf after {} gradient updates. Reduced cost-scaling factor to {}", total, costScalingFactor_);
    else
      LOG(debug, "Seen NaN/Inf after {} gradient updates, Reduced cost-scaling factor to minimum {}. Pruning NaNs now.", total, costScalingFactor_);
  }

  // Resetting counts after cost-scale change
  noNanSeen_ = 0;
  nanSeen_ = 0;
}

float GraphGroup::checkNanOrNorm(size_t i, size_t begin, size_t end) {
  auto curGrad = graphs_[i]->params()->grads()->subtensor(begin, end-begin);
  
  // If costScaling_ then check for NaN values if the costScalingFactor_ is larger than
  // the minimum. If a NaN value is seen we exit here and will reduce the factor next and 
  // this skips an update. 
  // If costScalingFactor_ is already at the minimum, prune the NaN values away. This replaces 
  // NaNs with 0. Updates are not skipped any more.
  // Regardless of NaNs, we clip +/-inf to the largest corresponding values for the gradient value type.
  // This changes the gradient but seems to be quite stable. In effect, for fp16 this is equivalent 
  // to gradient clipping at (65504.f / costScalingFactor_) which in most cases is still large. 
  if(costScaling_ || checkGradientNan_) {
    bool pruneNaN = !checkGradientNan_ && costScalingFactor_ == costScalingFactorMinimum_;
    bool clipInf  = !checkGradientNan_;
    bool saneGradient = SanitizeGradient(curGrad, graphs_[i]->allocator(), pruneNaN, clipInf);

    // This should never happen, if it does, something is wrong with the kernel above and needs to be fixed.
    ABORT_IF(pruneNaN && clipInf && !saneGradient, "We are removing NaNs and clipping Infs, but gradient is still not sane??");

    if(!saneGradient) {
      LOG(debug, "Found NaN");
      return std::numeric_limits<float>::quiet_NaN();
    }
  }

  // The optional clipping above will affect the norm here. The norm can be non-finite despite the above
  // gradient sanitization, hence check again and propagate a NaN.
  if(dynamicGradientScaling_) {
    auto gNorm = L2Norm(curGrad, graphs_[i]->allocator());
    if(isFinite(gNorm) && gNorm > 0.0)
      return gNorm;
    else 
      return std::numeric_limits<float>::quiet_NaN();
  }

  return 0.f;
};

float GraphGroup::executeAndCollectNorm(const std::function<float(size_t, size_t, size_t)>& task) {
  auto gradNorm = comm_->foreach(task, accNanOrNorm, 0.f);
  if(mpi_) { // accumulate gradientNorm from subprocesses
    auto gradNormSquared = gradNorm * gradNorm; // undo sqrt
    mpi_->allReduce(&gradNormSquared, &gradNormSquared, 1, MPI_FLOAT, MPI_SUM); // sum all
    
    if(shardingMode_ == ShardingMode::local) // we already have the correct norm on one device, but we also need to check for NaN
      gradNormSquared /= (float)mpi_->numMPIProcesses();
    
    gradNorm = std::sqrt(gradNormSquared); // redo sqrt
  }
  return gradNorm;
}

/**
 * This function computes are normalization factor that is applied to the gradient before an update.
 * Depending on various settings this will return a normalizer that can perform a combination of:
 * - apply a cost scaling factor if cost scaling is enabled
 * - normalize the gradient by the number of words in a batch if requested (turning ce-sum in to ce-mean). @TODO: once fp16 stability issues are proven to not be caused by this, remove.
 * - re-scale the gradient based on a dynamic running average of gradient norms
 */
float GraphGroup::computeNormalizationFactor(float gNorm, size_t updateTrgWords) {
  float normalizationFactor = 1.f;

  if(costScaling_)
    normalizationFactor *= costScalingFactor_;

  if(options_->get<bool>("normalize-gradient"))
    normalizationFactor *= updateTrgWords;

  if(!isFinite(gNorm)) // we are checking the sanity of the gradient elsewhere
    return normalizationFactor;
  
  if(dynamicGradientScaling_) {
    // make gradient norm invariant to changes in costScalingFactor_, luckily norm(c * g) = c * norm(g)
    if(costScaling_)
      gNorm = gNorm / costScalingFactor_;
    
    // Normalize gradient norm w.r.t. number of labels in batch for statistics, 
    // there should be no gradient normalization before this point, @TODO: check this
    gNorm = gNorm / updateTrgWords; 
    
    size_t window; float gNormAvgTransform, gNormVarTransform, gNormTransform, gNormAvg;
    if(dynamicGradientScalingUseLogs_) {
      // tracking the log of the gradient norms rather than the gradient norms itself results in a larger standard deviation as the actual
      // gradient norms go towards 0. From observation, the STD (of log norms) tends to become near constant after some time while the averages keep decreasing.
      std::tie(window, gNormAvgTransform, gNormVarTransform) = scheduler_->getLogGradientNormStats();
      gNormTransform = std::log(gNorm);             // we are using the average of log norms, so we need to compute the log
      gNormAvg       = std::exp(gNormAvgTransform); // for rescaling we need to undo the log, we assume avg(log(norm)) is roughly log(avg(norm))
    } else {
      std::tie(window, gNormAvgTransform, gNormVarTransform) = scheduler_->getGradientNormStats();
      gNormTransform = gNorm;              // we are not using logs, so we can just use the normal gradient norm
      gNormAvg       = gNormAvgTransform;  // we are getting the actual running average of gradient norms, no transformation needed.  
    }
    
    auto deltaTransform    = gNormTransform - gNormAvgTransform; // compute the difference between the current transformer gradient norm and the running average.
    auto gNormStdTransform = std::sqrt(gNormVarTransform);       // compute STD for the running average of (log) gradient norms.

    float fadeoutMultiplier = 1.f;
    if(dynamicGradientScalingFadeout_ > 0ul) // fade out linearly after that many updates @TODO: allow units other than updates
      fadeoutMultiplier = (float)std::max(dynamicGradientScalingFadeout_, scheduler_->numberOfBatches()) / (float)dynamicGradientScalingFadeout_;

    float dynamicGradientScalingFactorWithFadeout = dynamicGradientScalingFactor_ * fadeoutMultiplier; // if fadeoutMultiplier increases dynamic gradient scaling becomes less and less likely to happen over time.
    // delta of (log) gradient norm vs (log) gradient norm average is larger than N standard deviations
    // hence rescale gradient using the average.
    if(scheduler_->numberOfBatches() >= window && deltaTransform > dynamicGradientScalingFactorWithFadeout * gNormStdTransform) {
      if(isMainProcess())
        LOG(debug, "log gradient norms: {} :: {:.4f} - {:.4f} = {:.4f} > {:.4f} * {:.4f} - scaling gradient by {:.4f}",
            dynamicGradientScalingUseLogs_, gNormTransform, gNormAvgTransform, deltaTransform, dynamicGradientScalingFactorWithFadeout, gNormStdTransform, gNormAvg / gNorm);

      normalizationFactor *= gNorm / gNormAvg; // since we later do gradient / normalizationFactor this divides by norm and multiplies by the average, rescaling to the average. 
    }
  }

  return normalizationFactor;
};

void GraphGroup::load() {
  validate();
  auto scatterFn = [&](const io::Item& optimizerState, const OptimizerBase::ScatterStateSetFunc& setShardFn) {
    comm_->scatterState(optimizerState, setShardFn);
  };
  load(scatterFn);
}

void GraphGroup::save(bool isFinal) /*override*/ {
  auto gatherOpt  = [&](const OptimizerBase::GatherStateGetFunc& getShardFn) {
    return comm_->gatherState(getShardFn);
  };
  save(isFinal, gatherOpt);
}

void GraphGroup::load(const OptimizerBase::ScatterStateFunc& scatterFn) {
  /*
  if not no-reload (=> i.e. do reload):
    restore scheduler
    if checkpoint is available or not no-reload-checkpoint:
      reload from checkpoint
    else if model is available:
      reload from model, but warn that no checkpoint was used and the model could be smoothed
  else if pretrained-model path given:
    initialize matching weights from pretrained model
  else:
    (implicitly) don't do anything => initialize randomly later
  */
  if(!options_->get<bool>("no-reload")) {
    std::string modelFileName = options_->get<std::string>("model");

    if(filesystem::exists(modelFileName)) {
      if(scheduler_)
        scheduler_->load(modelFileName);
      // we just load it N times from disk (it'll be in disk cache after the first)
      // this also allocates memory correctly when calling forward() inside restoreFromCheckPoint
      size_t i = 0;
      for(auto graph : graphs_)
        models_[i++]->load(graph, modelFileName);

      // try to restore everything from checkpoint now
      restoreFromCheckpoint(modelFileName, scatterFn);
    } else if(options_->hasAndNotEmpty("pretrained-model")) {
      std::string nameInit = options_->get<std::string>("pretrained-model");
      LOG(info, "[training] Initializing model weights with pre-trained model {}", nameInit);

      size_t i = 0;
      for(auto graph : graphs_)
        models_[i++]->load(graph, nameInit, false);
    }
  }
}

bool GraphGroup::restoreFromCheckpoint(const std::string& modelFileName, 
                                       const OptimizerBase::ScatterStateFunc& scatterFn) {
  /*
  if model checkpoint is available:
    - load model from checkpoint, not from model.npz
    - abort if checkpoint model and graph size do not match, probably due to different model or precision
  */

  std::string checkpointName = modelFileName + ".optimizer.npz"; // @TODO: change to .checkpoint.npz, would break backwards compat

  if(!filesystem::exists(checkpointName)) {
    LOG(warn, "No checkpoint found, parameters reloaded from last inference model");
    return false; // failed to restore
  }

  auto items = io::loadItems(checkpointName);
  
  // make sure all nodes see the same checkpoint data, may not be the case with distributed file systems
  // when there was a delay in updating the caches accross nodes. So here node 0 sends its data to all.
  // We still load them all from disk, but that serves more as a trick to allocate the correct memory.
  if(mpi_)
    for(auto& item : items)
      mpi_->bCast(item);

  // @TODO: probably we want to have the list of DeviceIds as an attribute
  std::vector<Ptr<Backend>> backends;
  for(auto graph : graphs_)
    backends.push_back(graph->getBackend());
  optimizerShards_[0]->load(items, optimizerShards_, backends, scatterFn, isMainProcess());

  // restore the graph parameters from the checkpoint master copy.
  auto found = std::find_if(items.begin(), items.end(),
    [](const io::Item& item) { return item.name == "master_parameters"; });

  if(found == items.end()) {
    LOG(warn, "No master parameters found in checkpoint, parameters reloaded from last inference model");
    return false; // failed to restore
  }

  auto& masterParameters = *found;
  for(auto graph : graphs_) {
    graph->forward(); // allocate graph parameter memory and initialize parameters from inference model. This needs to
    // run a full forward pass over the paramters to allocate the parameters values in order (by parameter name).
    // Just doing graph->params()->allocateForward() is not sufficient.
    ABORT_IF(graph->params()->vals()->shape() != masterParameters.shape,
             "Graph parameter sizes and master copy parameter sizes in checkpoint do not match");

    // Convert type of io::Item to match graph parameter type.
    if(masterParameters.type != graph->params()->vals()->type())
      masterParameters.convert(graph->params()->vals()->type());

    graph->params()->vals()->set(masterParameters);
    graph->clear();
  }

  LOG(info, "[training] Master parameters and optimizers restored from training checkpoint {} and {}", modelFileName, checkpointName);
  return true; // succeeded to restore
}

void GraphGroup::saveCheckpoint(const std::string& modelFileName,
                                const OptimizerBase::GatherStateFunc& gatherFn) {
  // @TODO: change to .checkpoint.npz, would break backwards compat                                  
  std::string checkpointName = modelFileName + ".optimizer.npz";

  std::vector<io::Item> items;
  optimizerShards_[0]->save(items,
                            optimizerShards_,
                            gatherFn,
                            isMainProcess());
                            
  if(isMainProcess()) { // only main process does the actual saving
    auto found = std::find_if(items.begin(), items.end(),
      [](const io::Item& item) { return item.name == "master_parameters"; });

    if(found == items.end()) {
      // if the optimizer does not provide a master parameters copy (the default when training with full precision)
      // then dump the parameters of graphs_[0] into the checkpoint. This should be called when the original parameters
      // are in the graph, not the smoothed version. Here we are getting called after a double swap, so that should be
      // the case.
      io::Item masterParameters;
      graphs_[0]->params()->vals()->get(masterParameters, "master_parameters");
      items.push_back(masterParameters);
    }

    
    LOG(info, "[training] Saving training checkpoint to {} and {}", modelFileName, checkpointName);
    io::saveItems(checkpointName, items);
  }
}

void GraphGroup::save(bool isFinal,
                      const OptimizerBase::GatherStateFunc& gatherOptimizerStateFn) {
  barrier(); // (for better grouping of log messages)

  // bring the smoothed model in
  // Note that it is sharded. For multi-node, it is sharded over multiple machines, so this is a network access.
  // Also note that the swap must run on all MPI processes concurrently, although only one actually validates.

  swapWithSmoothed();
  
  if(isFinal && scheduler_)
    scheduler_->validate(graphs_, isFinal);

  barrier(); // (for better grouping of log messages)
  
  std::string modelFileName = options_->get<std::string>("model");
  if(isMainProcess()) {
    // save main model file
    if(options_->get<bool>("overwrite")) {
      models_[0]->save(graphs_[0], modelFileName, /*saveTranslatorConfig=*/true);
      // save scheduler-related state
      if(scheduler_)
        scheduler_->save(modelFileName);
    } else {
      if(!isFinal) { // save a model with iteration number
        std::string numberOfBatches = scheduler_ ? std::to_string(scheduler_->numberOfBatches()) : "unknown";
        std::string nameOverwrite = modelFileName;
        nameOverwrite.replace(modelFileName.size() - 4, 4, ".iter" + numberOfBatches + ".npz");
        models_[0]->save(graphs_[0], nameOverwrite);
      }
      models_[0]->save(graphs_[0], modelFileName, /*saveTranslatorConfig=*/true);

      // save scheduler-related state
      if(scheduler_)
        scheduler_->save(modelFileName);
    }
  }

  swapWithSmoothed();
  saveCheckpoint(modelFileName, gatherOptimizerStateFn);
  
  barrier(); // (for better grouping of log messages)
}

void GraphGroup::swapWithSmoothed() {
  auto swap = [&](size_t i, size_t begin, size_t end) {
    auto curParam = graphs_[i]->params()->vals()->subtensor(begin, end-begin);
    optimizerShards_[i]->swapWithSmoothed(curParam);
    return true; // dummy success
  };
  comm_->foreach(swap);
  comm_->allGatherParams();
  
  if(shardingMode_ == ShardingMode::local)
    comm_->broadcastParams();
    
  barrier();
}

void GraphGroup::validate() { //@TODO: rename this function to something less confusing.
  ABORT_IF(finalized_, "Training has already finished.");
}

void GraphGroup::finalize() {
  finalized_ = true;
}

/**
 * Determine maximal batch size that can fit into the given workspace
 * so that reallocation does not happen. Rather adjust the batch size
 * based on the stastistics collected here. Activated with
 * `--mini-batch-fit`.
 * In a multi-GPU scenario, the first GPU is used to determine the size.
 * The actual allowed size is then determined by multiplying it with the
 * number of devices, which is passed in as the 'multiplier'.
 */
// @TODO: Can this be made const? It seems wrong to have a stateful method that still returns a result.
Ptr<data::BatchStats> GraphGroup::collectStats(Ptr<ExpressionGraph> graph,
                                               Ptr<models::ICriterionFunction> model,
                                               const std::vector<Ptr<Vocab>>& vocabs,
                                               double multiplier) {
  // this runs with fake values, we do not care for overflow/underflow
  bool throwNan = graph->getThrowNaN();

  graph->setThrowNaN(false);

  auto stats = New<data::BatchStats>();
  size_t numFiles = numberOfInputFiles();

  // Initialize first batch to step size
  size_t first = options_->get<size_t>("mini-batch-fit-step");

  // Increase batch size and sentence length by this step size
  size_t step = options_->get<size_t>("mini-batch-fit-step");

  size_t maxLength = options_->get<size_t>("max-length");
  maxLength = (size_t)(std::ceil(maxLength / (float)step) * step);

  // this should be only one class label per line on input, hence restricting length to 1
  std::vector<size_t> localMaxes(numFiles, maxLength);
  auto inputTypes = options_->get<std::vector<std::string>>("input-types", {});
  for(int i = 0; i < inputTypes.size(); ++i)
    if(inputTypes[i] == "class")
      localMaxes[i] = 1;

  size_t maxBatch = 512;
  bool fits = true;
  while(fits) {
    std::vector<size_t> lengths(numFiles, first);

    for(int j = 0; j < lengths.size(); ++j) // apply length restrictions
      lengths[j] = std::min(lengths[j], localMaxes[j]);

    auto batch = data::CorpusBatch::fakeBatch(lengths, vocabs, maxBatch, options_);
    auto loss = model->build(graph, batch);
    fits = graph->fits();
    if(fits)
      maxBatch *= 2;
  }

  // Do a binary search for maxmimum batch size that fits into given workspace memory
  // for a tested sentence length.
  for(size_t i = step; i <= maxLength; i += step) {
    size_t start = 1;
    size_t end = maxBatch;

    std::vector<size_t> lengths(numFiles, i);
    for(int j = 0; j < lengths.size(); ++j)  // apply length restrictions
      lengths[j] = std::min(lengths[j], localMaxes[j]);
    fits = true;

    do {
      size_t current = (start + end) / 2;
      auto batch = data::CorpusBatch::fakeBatch(lengths, vocabs, current, options_);
      auto loss = model->build(graph, batch);
      fits = graph->fits();

      LOG(debug, "[batching] length: {} - size: {} - fits: {}", lengths[0], current, fits);

      if(fits) {
        stats->add(batch, multiplier);
        start = current + 1;
      } else {
        end = current - 1;
      }
    } while(end >= start);

    maxBatch = start;
  }

  // set back to original value for aborting on NaN or Inf
  graph->setThrowNaN(throwNan);

  return stats;
}

void GraphGroup::setTypicalTrgBatchWords(size_t typicalTrgBatchWords) { // needed for dynamic MB scaling
  typicalTrgBatchWords_ = (double)typicalTrgBatchWords;
}

double GraphGroup::getTypicalTrgBatchWords() {
  return typicalTrgBatchWords_;
}

void GraphGroup::updateAverageTrgBatchWords(size_t trgBatchWords) {
  typicalTrgBatchWords_ = 0.99 * typicalTrgBatchWords_ + 0.01 * (double)trgBatchWords; // record a running average of the batch size, factors are chosen empirically.
}

size_t GraphGroup::numberOfInputFiles() {
  if(options_->get<bool>("tsv", false)) {
    size_t n = options_->get<size_t>("tsv-fields");
    if(n > 0 && options_->get("guided-alignment", std::string("none")) != "none")
      --n;
    if(n > 0 && options_->hasAndNotEmpty("data-weighting"))
      --n;
    return n;
  }
  return options_->get<std::vector<std::string>>("train-sets").size();
}

}  // namespace marian
