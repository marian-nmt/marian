#pragma once

#if MPI_FOUND
#include "mpi.h"
#endif

#ifdef CUDA_FOUND
#include "cuda_runtime.h"
#endif

#include <condition_variable>
#include <future>
#include <thread>

#include <boost/filesystem.hpp>
#include <boost/thread/locks.hpp>
#include <boost/thread/shared_mutex.hpp>

#include "3rd_party/threadpool.h"
#include "training/graph_group.h"

namespace marian {

/**
 * Multi-node graph group for asynchronous training over multiple
 * machines each with one or multiple GPUs
 */
class MultiNodeGraphGroupSync : public GraphGroup {
public:
  virtual void setScheduler(Ptr<Scheduler> scheduler);

protected:
  ////////////////////////////////////////////////////////////////////////////
  // General variables.

  /** Number of clients on nodes in MPI world (cluster). */
  std::vector<int> numberClientsOfNodes_;  //@TODO not used for now, but might
                                           // be useful maybe?

  /** Whether graph group has been properly initialized with a first batch. */
  bool initialized_{false};

  /** Memory allocators for tensors (GPUs). */
  std::vector<Ptr<TensorAllocator>> allocators_;

  ////////////////////////////////////////////////////////////////////////////
  // Client variables.

  /** Thread pool to enable clients to run concurrently. */
  ThreadPool* clientThreadPool_;

  /** Graph builders for clients (which run forward and backward passes). */
  std::vector<Ptr<models::ModelBase>> clientBuilders_;

  /** Graphs of clients. */
  std::vector<Ptr<ExpressionGraph>> clientGraphs_;

  /** Devices (GPUs) on this node. */
  std::vector<size_t> devices_;

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

  /** MPI rank of this node. */
  int mpi_my_rank_{0};

  /** Number of nodes in MPI world (cluster). */
  int mpi_comm_world_size_{1};

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
  bool synchronization_happened{false};

  Ptr<OptimizerBase> syncOptimizer_;

  std::vector<std::mutex> optDelayMutex_;
  std::vector<size_t> delay_count;
  std::vector<int> totalBatchWords;
  std::vector<Tensor> accGradients, accGradientBuffer;

  bool movingAvg_{false};
  float mvDecay_{1e-4};

  /**
   * Allocate new tensor on given GPU and store allocator.
   */
  Tensor newTensor(int size, Ptr<Backend> backend);

  /*
   * exponential smoothing
   */
  void updateMovingAverage(Tensor paramsAvg, Tensor params, size_t batches);

  /**
   * Setup training environment and launch server thread and (if enabled) client
   * communication overlap threads..
   * Includes setting up MPI, node and shard sizes, clients, server shards and
   * communication overlap stuff.
   */
  virtual void init(Ptr<data::Batch> batch);

  /**
   * Setup MPI world size and rank of this node.
   */
  void setupMPI();

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

  /**
   * Load the GPU configuration of this node (i.e. which GPUs to use) and the
   * number of GPUs on the other nodes.
   */
  void loadDeviceConfig(std::vector<size_t> deviceConfig) {
    size_t index = 0, node = 0, nClientsSeen = 0;
    numberClientsOfNodes_ = std::vector<int>(mpi_comm_world_size_, 0);
    while(index < deviceConfig.size()) {
      if(numberClientsOfNodes_[node] == 0) {
        numberClientsOfNodes_[node] = deviceConfig[index];
        nClientsSeen = 0;
      } else if(nClientsSeen < numberClientsOfNodes_[node]) {
        if(node == mpi_my_rank_) {
          devices_.push_back(deviceConfig[index]);
        }
        nClientsSeen++;
      } else {
        node++;
        index--;
      }
      index++;
    }
  }

public:
  /**
   * (Constructor) Call super class and initialize client graphs and builders.
   */
  MultiNodeGraphGroupSync(Ptr<Config> options)
      : GraphGroup(options),
        tau_{options_->get<size_t>("optimizer-delay")},
        movingAvg_{options_->get<float>("exponential-smoothing") > 0},
        mvDecay_{options_->get<float>("exponential-smoothing")},
        syncOptimizer_{Optimizer(options_)} {
    // Set up devices for this node
    setupMPI();  // Setup MPI before creating device vectors
    std::vector<size_t> devices;
    for(auto& d : options_->getDevices())
      devices.push_back(d.no);
    loadDeviceConfig(devices);

    // Create builders and graphs for clients.
    for(size_t i = 0; i < devices_.size(); i++) {
      clientGraphs_.push_back(New<ExpressionGraph>());
      clientGraphs_[i]->setDevice({devices_[i], DeviceType::gpu});
      clientGraphs_[i]->reserveWorkspaceMB(options_->get<size_t>("workspace"));
      clientBuilders_.push_back(
          models::from_config(options_, models::usage::training));
    }
  }

  /**
   * (Destructor) Shut down server shard thread and (if comm. overlap enabled)
   * communication overlap threads.
   */
  virtual ~MultiNodeGraphGroupSync() {
    //@TODO merge with finalize method
    delete clientThreadPool_;
  }

  /**
   * Update any client model with given batch if batch is assigned to this node.
   */
  void update(Ptr<data::Batch> batch) {
    ABORT_IF(finalized_, "Training has already finished.");
    if(batchIter_ % mpi_comm_world_size_
       == mpi_my_rank_) {  // Only take batch assigned to this node
      execute(batch);
    }
    batchIter_++;
  }

  /**
   * Load models from disk if file exists and setting is not disabled
   */
  void load() {
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
   */
  void save(bool final = false) { save(clientGraphs_[0], final); }

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
  Ptr<data::BatchStats> collectStats() {
    return GraphGroup::collectStats(
        clientGraphs_[0], clientBuilders_[0], devices_.size());
  }

  virtual void finalize() {
    finalized_ = true;
#if MPI_FOUND
    MPI_Finalize();
#endif
  }
};
}
