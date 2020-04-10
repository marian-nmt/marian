#pragma once

#include "training/graph_group.h"
#include "training/communicator.h"
#include "3rd_party/threadpool.h"

#include <condition_variable>
#include <future>
#include <thread>

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
   * Batch number counter used for evenly distributing mini-batches across
   * nodes.
   */
  size_t batchIter_ = 0;

  ////////////////////////////////////////////////////////////////////////////
  // Communication variables.

  /** Number of clients on nodes in MPI world (cluster). */
  std::vector<int> numberClientsOfNodes_;  //@TODO not used for now, but might
                                           // be useful maybe?

  /**
   * Variables for optimizer delay and synchronous SGD
   */
  size_t tau_{1};
  std::mutex sumGradientMutex_;
  std::mutex updateParamsMutex_;
  std::mutex sumCostMutex_;
  Tensor accGradientsSync;
  Tensor sumGradientBuffer;
  Tensor paramsAvg_;
  std::vector<float> accGradientsSync_cpu;
  std::vector<float> receiveBuffer_cpu;

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
  void sendReceiveUpdateSync();

  void execute(Ptr<data::Batch> batch);

public:
  /**
   * (Constructor) Call super class and initialize client graphs and builders.
   */
  MultiNodeGraphGroupSync(Ptr<Options> options, Ptr<IMPIWrapper> mpi)
      : Base(options, mpi),
        tau_{(size_t)options_->get<double>("optimizer-delay")},
        syncOptimizer_{Optimizer(options_)},
        movingAvg_{options_->get<float>("exponential-smoothing") > 0},
        mvDecay_{options_->get<float>("exponential-smoothing")} {
  }

  /**
   * Update any client model with given batch if batch is assigned to this node.
   */
  void update(Ptr<data::Batch> batch) override {
    validate();
    if(batchIter_ % mpi_->numMPIProcesses() == mpi_->myMPIRank()) {  // Only take batch assigned to this node
      execute(batch);
    }
    batchIter_++;
  }

  /**
   * Load models from disk if file exists and setting is not disabled
   */
  void load() override {
    if(!options_->get<bool>("no-reload")) {
      std::string name = options_->get<std::string>("model");

      if(filesystem::exists(name)) {
        if(scheduler_)
          scheduler_->load(name);
        size_t i = 0;
        for(auto graph : clientGraphs_)
          clientBuilders_[i++]->load(graph, name);
      } else if(options_->hasAndNotEmpty("pretrained-model")) {
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
   */
  void save(bool final = false) override { save(clientGraphs_[0], final); }

  /**
   * Save model of given graph to disk.
   */
  void save(Ptr<ExpressionGraph> graph, bool final = false) {
    int idx = 0;
    for(int i = 0; i < clientGraphs_.size(); ++i) {
      if(graph == clientGraphs_[i]) {
        idx = i;
        break;
      }
    }

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

  /**
   * Collect statistics from first client's graph.
   */
  Ptr<data::BatchStats> collectStats(const std::vector<Ptr<Vocab>>& vocabs) {
    return GraphGroup::collectStats(
        clientGraphs_[0], clientBuilders_[0], vocabs, (double)devices_.size());
  }
};
}  // namespace marian
