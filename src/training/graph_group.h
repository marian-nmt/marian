#pragma once

#include "common/definitions.h"
#include "common/options.h"
#include "data/batch_generator.h"
#include "graph/expression_graph.h"
#include "models/model_base.h"
#include "optimizers/optimizers.h"
#include "training/scheduler.h"
#include "training/communicator.h"

namespace marian {

/**
 *  Base class for managing the training process across one, multiple gpus,
 *  or even multiple machines with multiple gpus.
 */
class GraphGroup {
protected:
  Ptr<Options> options_;
  Ptr<OptimizerBase> opt_;   // the optimizer
  Ptr<Scheduler> scheduler_; // scheduler that keeps track of how much has been processed
  bool finalized_{false};    // 'true' if training has completed (further updates are no longer allowed)
  size_t typicalTrgBatchWords_{ 0 }; // for dynamic batch sizing: typical batch size in words

public:
  GraphGroup(Ptr<Options> options) : options_(options), opt_(Optimizer(options)) {}

  virtual ~GraphGroup() {}

  virtual void update(Ptr<data::Batch> batch) = 0;

  virtual void load() = 0;

  virtual void save(bool isFinal = false) = 0;

  void validate() {
    ABORT_IF(finalized_, "Training has already finished.");
  }

  virtual void finalize() {
    finalized_ = true;
  }

  virtual void setScheduler(Ptr<Scheduler> scheduler) = 0;

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
  Ptr<data::BatchStats> collectStats(Ptr<ExpressionGraph> graph,
                                     Ptr<models::ICriterionFunction> model,
                                     const std::vector<Ptr<Vocab>>& vocabs,
                                     double multiplier = 1.) {
    auto stats = New<data::BatchStats>();

    size_t numFiles = options_->get<bool>("tsv", false)
                          ? options_->get<size_t>("tsv-fields")
                          : options_->get<std::vector<std::string>>("train-sets").size();

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
      auto cost = model->build(graph, batch);
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
        auto cost = model->build(graph, batch);
        fits = graph->fits();

        LOG(debug, "[batching] length: {} - size: {} - fits: {}", lengths[0], current, fits);

        if(fits) {
          stats->add(batch, multiplier);
          start = current + 1;
        } else {
          end = current - 1;
        }
      } while(end - start > step);

      maxBatch = start;
    }
    return stats;
  }

  void setTypicalTrgBatchWords(size_t typicalTrgBatchWords) { // needed for dynamic MB scaling
    typicalTrgBatchWords_ = typicalTrgBatchWords;
  }
};

/**
 *  Base class for multi-node versions of GraphGroups.
 */
class MultiNodeGraphGroupBase : public GraphGroup {
  using Base = GraphGroup;

protected:
  Ptr<IMPIWrapper> mpi_; // all MPI-like communication goes through this

  /** Devices (GPUs) on this node. */
  std::vector<size_t> devices_; // [num local GPUs]

  /** Graph builders for clients (which run forward and backward passes). */
  std::vector<Ptr<models::ICriterionFunction>> clientBuilders_;

  /** Graphs of clients. One entry per GPU on this node. */
  std::vector<Ptr<ExpressionGraph>> clientGraphs_; // [num local GPUs]

public:
  MultiNodeGraphGroupBase(Ptr<Options> options, Ptr<IMPIWrapper> mpi)
    : Base(options), mpi_(mpi) {

    // Set up devices for this node
    std::vector<size_t> devices; // set of GPU device ids for this MPI process
    for (auto& d : Config::getDevices(options_))
      devices.push_back(d.no);
    loadDeviceConfig(devices); // set up numberClientsOfNodes_[] and devices_[]

    // Create builders and graphs for clients; that is, for each GPU we use on this node.
    for (size_t i = 0; i < devices_.size(); i++) {
      clientGraphs_.push_back(New<ExpressionGraph>());
      clientGraphs_[i]->setDevice({ devices_[i], DeviceType::gpu });
      clientGraphs_[i]->reserveWorkspaceMB(options_->get<size_t>("workspace"));
      clientBuilders_.push_back(models::createCriterionFunctionFromOptions(options_, models::usage::training));
    }
  }

  /**
   * Load the GPU configuration of this node (i.e. which GPUs to use) and the
   * number of GPUs on the other nodes.
   */
  // deviceConfig has this format
  //  - for each node
  //     - number of GPUs on that node
  //     - GPU ids for that node
  // e.g. 0:0 1 1: 2 3 -> (2, (0, 1)) (2, (2,3))
  void loadDeviceConfig(std::vector<size_t> deviceConfig) {
    // parse device config array
    size_t index = 0; // cursor for next()
    auto next = [&]() { // helper function to get the next item
      ABORT_IF(index == deviceConfig.size(), "mal-formed device config array??");
      return deviceConfig[index++];
    };
    std::vector<std::vector<size_t>> allDevices(mpi_->numMPIProcesses());
    for (auto& devices : allDevices) {
      devices.resize(next());
      for (auto& device : devices)
        device = next();
    }
    ABORT_IF(index != deviceConfig.size(), "mal-formed device config array??");

    // validate
    ABORT_IF(allDevices.front().size() == 0, "no devices specified??");
    for (auto& devices : allDevices) {
      ABORT_IF(devices.size() != allDevices.front().size(), "all MPI nodes must use the same number of devices");
    }

    // get our own config
    devices_ = allDevices[mpi_->myMPIRank()];

    // log
    LOG(info, "[mpi rank {}] device configuration", mpi_->myMPIRank());
    for (auto& device : devices_)
      LOG(info, "[mpi rank {}]  - {}", mpi_->myMPIRank(), device);
  }

  virtual void finalize() override {
    if (mpi_)
      finalizeMPI(std::move(mpi_));
    Base::finalize();
  }
};
}  // namespace marian
