#pragma once

#if MPI_FOUND
#include "mpi.h"
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
class MultiNodeGraphGroup : public GraphGroup {
public:
  virtual void setScheduler(Ptr<Scheduler> scheduler);

protected:
  ////////////////////////////////////////////////////////////////////////////
  // General variables.

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
  // Server (shard) variables.

  /**
   * Main server thread that continually receives gradients from a client,
   * copies them to the shards on this node, runs the shard optimizers and then
   * returns the result to that client.
   */
  std::thread* serverShardThread_;

  /**
   * Main server CPU buffer to enable MPI sending and receiving (GPU -> CPU ->
   * Network -> CPU -> GPU).
   */
  std::vector<float> serverShardBufferCPU_;

  /**
   * Parts of the global parameters that are assigned to server shards on this
   * node.
   */
  std::vector<Tensor> shardParams_;

  /**
   * GPU buffer to store gradients received by this node's server thread, so
   * that they can be applied to the shard parameters.
   */
  std::vector<Tensor> shardGrads_;

  /**
   * Server shard optimizers used to update global parameters with gradients
   * received from clients.
   */
  std::vector<Ptr<OptimizerBase>> shardOptimizers_;

  /** Mutex to enforce critical sections in server shards. */
  std::vector<std::mutex> shardMutex_;

  ////////////////////////////////////////////////////////////////////////////
  // Communication variables.

  /** Number of clients on nodes in MPI world (cluster). */
  std::vector<int> numberClientsOfNodes_;

  /** Number of parameters allocated (sharded) to nodes. */
  std::vector<size_t> nodeSizes_;

  /** Number of parameters allocated to shards on THIS node. */
  std::vector<size_t> shardSizes_;

  /**
   * CPU buffer for sending gradients and receiving parameters via MPI.
   */
  std::vector<std::vector<float>> clientCommBuffersCPU_;

  /** MPI rank of this node. */
  int mpi_my_rank_{0};

  /** Number of nodes in MPI world (cluster). */
  int mpi_comm_world_size_{1};

  /**
   * Flag to indicate that an MPI message contains message info 
   * before sending the gradient (client -> server).
   */
  static const int MPI_TAG_GRAD_PUSH_MSG_{0};

  /**
   * Flag to indicate that an MPI message contains gradient (client -> server).
   */
  static const int MPI_TAG_GRAD_PUSH_{5};

  /**
   * Flag to indicate that an MPI message contains parameters (server ->
   * client).
   */
  static const int MPI_TAG_PARAM_PUSH_{10};

  /**
   * Message info indices: 0 = size; 1 = originating client; 2 = number of batch
   * words; 3 = status of node
   */
  static const unsigned int MSG_INFO_SIZE_{0}, MSG_INFO_CLIENT_{1},
      MSG_INFO_BATCHWORDS_{2}, MSG_INFO_STATUS_{3};

  /**
   * Status of node: 0 = training; 1 = finished. Used to indicate to other nodes
   * whether this node is still training.
   */
  static const unsigned int STATUS_NODE_TRAINING_{0}, STATUS_NODE_FINISHED_{1};

  /**
   * Whether client computations should continue while gradients and parameters
   * are being exchanged with server shards.
   */
  bool clientCommOverlap;

  ////////////////////////////////////////////////////////////////////////////
  // Overlapping communication and computation variables.

  /**
   * Threads for client communication overlap. They send gradients and receive
   * gradients to/from the server shards when the communication buffer is
   * filled.
   */
  std::vector<std::thread*> clientCommThreads_;

  /**
   * Flags indicating whether the client overlapping communication threads
   * should stop.
   */
  bool stopClientCommThreads_{false};

  /**
   * GPU buffer (tensor) to sum up gradients computed by the clients. Used to
   * enable clients to proceed with computations while waiting for communication
   * channel to become available.
   */
  std::vector<Tensor> clientSummedGradsGPU;

  /** Summed word counts of clients. Used for overlap purposes. */
  std::vector<size_t> clientSummedWordCounts_;

  /**
   * Word counts of clients submitted to overlapping/communication thread for
   * current mini-batch.
   */
  std::vector<size_t> clientCommittedWordCounts_;

  /**
   * Optimizers used locally by clients to apply gradients from the
   * communication thread to their graph's parameters.
   */
  std::vector<Ptr<OptimizerBase>> clientLocalOptimizers_;

  /**
   * GPU buffers used by clients to copy gradients to for use by the
   * communication threads.
   */
  std::vector<Tensor> clientCommOverlapBuffersGPU_;

  /**
   * Flags to indicate whether a client's communication buffer is filled, in
   * which case the communication thread will exchange the data with server
   * shards.
   */
  std::vector<bool> clientCommOverlapBuffersFilled_;

  /**
   * Mutex to enable communication thread to wait for its buffers to be filled.
   */
  std::vector<std::mutex> mutexClientCommOverlapBuffersFilled_;

  /**
   * Condition variable to notify communication threads that their buffers have
   * been filled.
   */
  std::vector<std::condition_variable> cvClientCommOverlapBuffersFilled_;

  /**
   * Variables for optimizer delay
   */
  size_t tau_{1};
  std::vector<std::mutex> optDelayMutex_;
  std::vector<size_t> delay_count;
  std::vector<int> totalBatchWords;
  std::vector<Tensor> accGradients, accGradientBuffer;

  /**
   * LocalOptimizers related variables
   */
  bool useLocalOpt_;

  /**
   * Allocate new tensor on given GPU and store allocator.
   */
  Tensor newTensor(int size, Ptr<Backend> backend);

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
   * Calculate the size of each node in the MPI world (cluster).
   * Account for the edge case where the last node has fewer parameters because
   * the model size is not perfectly divisible by the number of nodes.
   */
  void calculateNodeSizes();

  /**
   * Initialize a CPU buffer for each client on this node for storing gradients
   * or parameters.
   * Required for sending GPU data through MPI to other nodes (GPU -> CPU -> MPI
   * network).
   */
  void initClientCpuBuffers();

  /**
   * Initialize variables required for overlapping client computations and
   * communication.
   * Includes summed and committed word counts, buffer flags, mutexes and
   * condition variables.
   */
  void initClientCommOverlapVars();

  /**
   * Initialize GPU tensors required for overlapping client computations and
   * communication.
   * Includes secondary buffers for params/grads, buffers for locally summing
   * gradients, and local optimizers to apply received gradients to client
   * parameters.
   */
  void initClientCommOverlapGpuTensors();

  /**
   * Setup server shards that will receive gradients from clients, apply them to
   * their part of the global parameters, and send them back to the same
   * clients.
   * There is one server shard per GPU. (Each GPU acts both as a client and as a
   * server shard.)
   */
  void setupServerShards();

  /**
   * Calculate the size of each shard on this node.
   * Account for the edge case where the last shard has fewer parameters because
   * the node size is not perfectly divisibly by the number of shards.
   */
  void calculateShardSizes();

  /**
   * Initialize the GPU tensors for storing the parameters and gradients of each
   * server shard.
   */
  void initShardGpuTensors();

  /**
   * Launch independent thread which continually receives gradients assigned to
   * this shard from any client, runs the shard optimizer and sends back the
   * updated parameters.
   */
  virtual void launchServerThread();

  /**
   * Safely shut down the launched server shard thread.
   */
  void shutDownServerThread();

  /**
   * Launch independent threads which continually synchronize their client's
   * gradients/parameters whenever the respective communication buffers are
   * full.
   */
  void launchCommOverlapThreads();

  /**
   * Safely shut down the launched communication overlap threads
   */
  void shutDownCommOverlapThreads();

  /**
   * Send new gradients to the server shards and receive the updated (global)
   * parameters.
   *
   * @param newGrads Gradients to send
   * @param oldParams Parameters to replace
   * @param gpu GPU/client performing synchronize (to access appropriate buffers
   *  etc.)
   * @param batchWords Number of batch words to pass to server shard optimizers
   */
  virtual void synchronizeWithServerShards(Tensor newGrads,
                                           Tensor oldParams,
                                           int gpu,
                                           size_t batchWords = 0);

  /**
   * Execute given batch on this node, pushing/pulling the resulting
   * gradients/parameters to/from the server shards
   * or -- if comm. overlap enabled -- to/from the communication buffers,
   * summing gradients locally if the communication thread is busy
   *
   * @param batch Batch on which to perform forward and backward passes.
   */
  void execute(Ptr<data::Batch> batch);

  /**
   * Notify server shards that this node has finished training.
   */
  virtual void signalFinishedToServerShards();

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
  MultiNodeGraphGroup(Ptr<Config> options)
      : GraphGroup(options),
        tau_{options_->get<size_t>("optimizer-delay")},
        useLocalOpt_{options_->get<bool>("multi-node-local-optimizers")},
        clientCommOverlap{options_->get<bool>("multi-node-overlap")} {
    // Set up devices for this node
    setupMPI(); //Setup MPI before creating device vectors
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
  virtual ~MultiNodeGraphGroup() {
    if(initialized_) {
      if(clientCommOverlap) {
        shutDownCommOverlapThreads();
      }
      signalFinishedToServerShards();  // notify other nodes that this node has
                                       // finished training
      shutDownServerThread();
    }
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
    return GraphGroup::collectStats(clientGraphs_[0], clientBuilders_[0]);
  }

  virtual void finalize() {
    finalized_ = true;
  }
};
}
