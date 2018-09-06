#pragma once

// @TODO: Does this need to be a header at all? We can inline the entire class definition if we just add a factory function.

#include "training/graph_group.h"
#include "training/communicator.h"

#ifdef CUDA_FOUND
#include "cuda_runtime.h"
#endif

#include <future>
#include <thread>

#include <boost/filesystem.hpp>
#include <boost/thread/locks.hpp>
#include <boost/thread/shared_mutex.hpp>

#include "3rd_party/threadpool.h"

namespace marian {

/**
 * Multi-node graph group for asynchronous training over multiple
 * machines each with one or multiple GPUs
 */
class MultiNodeGraphGroupSync : public MultiNodeGraphGroupBase {
  using Base = MultiNodeGraphGroupBase;
public:
  virtual void setScheduler(Ptr<Scheduler> scheduler) override;

private:
  ////////////////////////////////////////////////////////////////////////////
  // General variables.

  /** Whether graph group has been properly initialized with a first batch. */
  bool initialized_{false};

  /** Memory allocators for tensors (GPUs). */
  std::vector<Ptr<TensorAllocator>> allocators_;

  ////////////////////////////////////////////////////////////////////////////
  // Client variables.

  /** Mutex to ensure clients are uniquely assigned to graphs and builders. */
  std::mutex mutexClientInit_;

  /** Mutex to avoid race conditions in scheduler. */
  std::mutex schedulerMutex_;

  /**
   * Global batch counter used for evenly distributing mini-batches across
   * nodes.
   * Global means that on all workers, this batch id refers to the same batch,
   * while each worker only processes a subset of batches.
   * Nodes process batches round-robin. Specifically, each node processes
   * the subset of batches with batchIter_ % mpi_->commWorldSize() == mpi_->myRank()).
   * @TODO: This is bad. The batches should be global and split into sub-batches across nodes.
   *        Otherwise batch ids are not comparable.
   */
  size_t batchIter_ = 0;

  ////////////////////////////////////////////////////////////////////////////
  // Communication variables.

  Ptr<Communicator> comm_;

  /**
   * Variables for optimizer delay and synchronous SGD
   */
  size_t tau_{1};
  std::mutex sumGradientMutex_;
  std::mutex updateParamsMutex_;
  std::mutex sumCostMutex_;
  Tensor accGradient_; // @TODO: which mutex guards this? Group variables by guarding mutexes
  Tensor sumGradientBuffer_; // buffer owned by sumGRAD
  Tensor paramsAvg_;
  std::vector<float> accGradientsSync_cpu;
  std::vector<float> receiveBuffer_cpu;
  bool synchronization_happened{false};

  Ptr<OptimizerBase> syncOptimizer_;

  std::vector<std::mutex> optDelayMutex_;
  std::vector<size_t> delay_count;
  std::vector<int> totalBatchWords;
  std::vector<Tensor> accGradients, accGradientBuffer;

  bool movingAvg_{false};
  float mvDecay_{1e-4f};

  /**
   * Allocate new tensor on given GPU and store allocator.
   */
  Tensor newTensor(int size, Ptr<Backend> backend);

  /*
   * exponential smoothing
   */
  void updateAvgParams(Tensor paramsAvg, Tensor params, size_t batches);

  /**
   * Setup training environment and launch server thread and (if enabled) client
   * communication overlap threads..
   * Includes setting up MPI, node and shard sizes, clients, server shards and
   * communication overlap stuff.
   */
  virtual void init(Ptr<data::Batch> batch);

  /**
   * Setup clients that will compute gradients and communicate them with the
   * server shards.
   * There is one client per GPU.
   */
  void setupClients(Ptr<data::Batch> batch);

  /**
   * Initialize the graphs (models) of all clients on this node with the given
   * batch.
   */
  void runBatchThroughClientGraphs(Ptr<data::Batch> batch);

  /**
   * Initialize the CPU arrays, with pinned memory for faster CudaMemCpy
   * operations.
   */
  void initCPUArrays();

  /**
   * Sums the gradients from a node, taking care of locking
   * @param gradient - the gradient
   */

  void sumGRAD(Tensor gradient);

  /**
   * Does the MPI Communication, parameter update and copying back parameters.
   * @TODO ALHAM. God function too godly?
   */
  void sendReceiveUpdateSync(Tensor accGradient);

  void execute(Ptr<data::Batch> batch);

public:
  /**
   * (Constructor) Call super class and initialize client graphs and builders.
   */
  MultiNodeGraphGroupSync(Ptr<Config> options)
      : Base(options),
        tau_{options_->get<size_t>("optimizer-delay")}, // do cross-node aggregation only every tau_ updates (defaults to 1)
        movingAvg_{options_->get<float>("exponential-smoothing") > 0}, // @TODO: redundant
        mvDecay_{options_->get<float>("exponential-smoothing")},
    syncOptimizer_{ Optimizer(options_) } { // @BUGBUG? Do we really have two optimizers?
    //comm_ = createCommunicator(clientGraphs_, options_->get<bool>("no-nccl", false));
  }

  /**
   * Update any client model with given batch if batch is assigned to this node.
   */
  void update(Ptr<data::Batch> batch) override {
    ABORT_IF(finalized_, "Training has already finished.");
    if(batchIter_ % mpi_->commWorldSize()
       == mpi_->myRank()) {  // Only take batch assigned to this node
      execute(batch);
    }
    batchIter_++;
  }

  /**
   * Load models from disk if file exists and setting is not disabled
   * @TODO: How is this specific to multi-node? This a general operation, no? Code dup
   */
  void load() override {
    if(!options_->get<bool>("no-reload")) {
      std::string name = options_->get<std::string>("model");

      if(boost::filesystem::exists(name)) {
        if(scheduler_)
          scheduler_->load(name);
        size_t i = 0;
        for(auto graph : clientGraphs_)
          clientBuilders_[i++]->load(graph, name);
      } else if(options_->has("pretrained-model")) {
        std::string init = options_->get<std::string>("pretrained-model");
        LOG(info,
            "Initialize model weights with the pre-trained model {}",
            init);
        size_t i = 0;
        for(auto graph : clientGraphs_)
          clientBuilders_[i++]->load(graph, init, false);
      }
    }
  }

  /**
   * Save model of first client's graph to disk
   * Only MPI node[0] saves the model.
   */
  void save(bool final = false) override {
    if (mpi_->myRank() == 0)
      saveGraph(clientGraphs_[0], final);
  }

private:
  /**
   * Save model of given graph to disk.
   */
  void saveGraph(Ptr<ExpressionGraph> graph, bool final = false) {
    // recover which client (device) owns this graph
    int idx = 0;
    for(int i = 0; i < clientGraphs_.size(); ++i) {
      if(graph == clientGraphs_[i]) {
        idx = i;
        break;
      }
    }

    // @TODO: This code does not seem specific to multi-node. Remove code dup.
    if(options_->get<bool>("overwrite")) {
      std::string name = options_->get<std::string>("model");

      clientBuilders_[idx]->save(clientGraphs_[idx], name, true);
      if(scheduler_)
        scheduler_->save(name);
    } else {
      std::string name = options_->get<std::string>("model");

      if(!final) {
        std::string numberOfBatches
            = scheduler_ ? std::to_string(scheduler_->numberOfBatches())
                         : "unknown";
        std::string nameOverwrite = name;
        nameOverwrite.replace(
            name.size() - 4, 4, ".iter" + numberOfBatches + ".npz");
        clientBuilders_[idx]->save(clientGraphs_[idx], nameOverwrite);
      }

      clientBuilders_[idx]->save(clientGraphs_[idx], name, true);
      if(scheduler_)
        scheduler_->save(name);
    }
  }
public:

  /**
   * Collect statistics from first node's first device's graph.
   * The total size per node is generalized by multiplying with the number of devices
   * The resulting statistics from the first node is then broadcast to the other nodes.
   * This assumes that all GPUs are of the same size.
   */
  Ptr<data::BatchStats> collectStats() {
    mpi_->barrier();
    // determine the statistics for one device
    std::vector<size_t> flattenedStats;
    if (mpi_->myRank() == 0) { // on node 0
      auto stats = GraphGroup::collectStats(
        clientGraphs_[0], clientBuilders_[0], devices_.size()); // @TODO: * tau_ ?
      flattenedStats = stats->flatten();
    }
    mpi_->bCast(flattenedStats, 0); // broadcast to all
    return New<data::BatchStats>(flattenedStats); // now all have the same BatchStats
  }
};
}  // namespace marian
