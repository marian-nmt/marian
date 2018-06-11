#include "training/graph_group_multinode.h"
#include "functional/functional.h"
#include "tensors/tensor_operators.h"

namespace marian {

/**
 * Set given scheduler to register training observers on the shard optimizers.
 */
void MultiNodeGraphGroup::setScheduler(Ptr<Scheduler> scheduler) {
  scheduler_ = scheduler;
  // optimizer has to be registered last to see a change of learning rate
  scheduler_->registerTrainingObserver(scheduler_);

  for(auto opt : shardOptimizers_) {
    scheduler_->registerTrainingObserver(opt);
  }
}

/**
 * Allocate new tensor on given GPU and store allocator.
 */
Tensor MultiNodeGraphGroup::newTensor(int size, Ptr<Backend> backend) {
  Tensor t;
  Ptr<TensorAllocator> allocator = New<TensorAllocator>(backend);
  allocator->reserveExact(size * sizeof(float));
  allocator->allocate(t, {1, size});
  allocators_.push_back(allocator);
  return t;
}

/**
 * Setup training environment and launch server thread and (if enabled) client
 * communication overlap threads.
 * Includes setting up MPI, node and shard sizes, clients, server shards and
 * communication overlap stuff.
 */
void MultiNodeGraphGroup::init(Ptr<data::Batch> batch) {
  // Setup clients and shards
  setupClients(batch);
  setupServerShards();
  if(clientCommOverlap) {
    initClientCommOverlapVars();
    initClientCommOverlapGpuTensors();
  }
  // Launch threads
  launchServerThread();  // For receiving and processing gradients and sending
                         // back parameters
  if(clientCommOverlap) {
    launchCommOverlapThreads();  // For communicating with server shards while
                                 // other threads do computations
  }

  // setup delayed gradient storage
  if(tau_ > 1) {
    delay_count = std::vector<size_t>(mpi_comm_world_size_);
    totalBatchWords = std::vector<int>(mpi_comm_world_size_);
    optDelayMutex_ = std::vector<std::mutex>(mpi_comm_world_size_);

    for(int i = 0; i < mpi_comm_world_size_; i++) {
      // Shard buffers across GPUs
      auto backend = clientGraphs_[i % devices_.size()]->getBackend();
      Tensor accGrad = newTensor(nodeSizes_[i], backend);
      Tensor accGradBuff = newTensor(nodeSizes_[i], backend);
      accGradients.push_back(accGrad);
      accGradientBuffer.push_back(accGradBuff);
    }
  }
}

/**
 * Setup MPI world size and rank of this node.
 */
void MultiNodeGraphGroup::setupMPI() {
#if MPI_FOUND
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_comm_world_size_);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_my_rank_);
#endif
}

/**
 * Setup clients that will compute gradients and communicate them with the
 * server shards.
 * There is one client per GPU.
 */
void MultiNodeGraphGroup::setupClients(Ptr<data::Batch> batch) {
  runBatchThroughClientGraphs(batch);
  calculateNodeSizes();
  initClientCpuBuffers();
  if(clientCommOverlap) {
    initClientCommOverlapVars();
    initClientCommOverlapGpuTensors();
  }
  clientThreadPool_ = new marian::ThreadPool(devices_.size(), devices_.size());
}

/**
 * Initialize the graphs (models) of all clients on this node with the given
 * batch.
 */
void MultiNodeGraphGroup::runBatchThroughClientGraphs(Ptr<data::Batch> batch) {
  for(int i = 0; i < devices_.size(); i++) {
    THREAD_GUARD(clientBuilders_[i]->build(clientGraphs_[i], batch);
                 clientGraphs_[i]->forward();
                 clientGraphs_[i]->getBackend()->synchronize(););
  }
}

/**
 * Calculate the size of each node in the MPI world (cluster).
 * Account for the edge case where the last node has fewer parameters because
 * the model size is not perfectly divisible by the number of nodes.
 */
void MultiNodeGraphGroup::calculateNodeSizes() {
  size_t modelSize = clientGraphs_[0]->params()->vals()->size();
  size_t nodeSize = ceilf(((float)modelSize) / mpi_comm_world_size_);
  for(int node = 0; node < mpi_comm_world_size_; node++) {
    size_t remainingModelSize = modelSize - (nodeSize * node);
    // Takes care of edge case where last node is smaller than the others
    nodeSizes_.push_back(std::min(nodeSize, remainingModelSize));
  }
}

/**
 * Initialize a CPU buffer for each client on this node for storing gradients or
 * parameters.
 * Required for sending GPU data through MPI to other nodes (GPU -> CPU -> MPI
 * network).
 */
void MultiNodeGraphGroup::initClientCpuBuffers() {
  // Initialize CPU buffers used to send GPU data through MPI (can't send
  // directly from GPUs)
  for(int i = 0; i < devices_.size(); i++) {
    // @TODO Optimization: Use full size to copy in one go, then send gradients
    // and receive parameters in parallel
    size_t size = nodeSizes_[mpi_my_rank_];
    clientCommBuffersCPU_.push_back(std::vector<float>(size));
  }
}

/**
 * Initialize variables required for overlapping client computations and
 * communication.
 * Includes summed and committed word counts, buffer flags, mutexes and
 * condition variables.
 */
void MultiNodeGraphGroup::initClientCommOverlapVars() {
  clientSummedWordCounts_ = std::vector<size_t>(devices_.size(), 0);
  clientCommittedWordCounts_ = std::vector<size_t>(devices_.size(), 0);
  clientCommOverlapBuffersFilled_ = std::vector<bool>(devices_.size(), false);
  mutexClientCommOverlapBuffersFilled_
      = std::vector<std::mutex>{devices_.size()};
  cvClientCommOverlapBuffersFilled_
      = std::vector<std::condition_variable>(devices_.size());
}

/**
 * Initialize GPU tensors required for overlapping client computations and
 * communication.
 * Includes secondary buffers for params/grads, buffers for locally summing
 * gradients, and local optimizers to apply received gradients to client
 * parameters.
 */
void MultiNodeGraphGroup::initClientCommOverlapGpuTensors() {
  size_t modelSize = clientGraphs_[0]->params()->vals()->size();
  for(int client = 0; client < devices_.size(); client++) {
    // Communication overlap buffer (for grads + params)
    Tensor commOverlapBuffer
        = newTensor(modelSize, clientGraphs_[client]->getBackend());
    commOverlapBuffer->copyFrom(clientGraphs_[0]->params()->vals());
    clientCommOverlapBuffersGPU_.push_back(commOverlapBuffer);
    // Gradients local sum buffer
    Tensor sumGrads = newTensor(modelSize, clientGraphs_[client]->getBackend());
    sumGrads->set(0);
    clientSummedGradsGPU.push_back(sumGrads);
    // Local optimizer to apply summed gradients
    clientLocalOptimizers_.push_back(Optimizer(options_));
    // => for simple SGD opt:
    // clientLocalOptimizers_.push_back(Optimizer<Sgd>(0.0001,
    // keywords::clip=Clipper<Norm>(1)));
  }
}

/**
 * Setup server shards that will receive gradients from clients, apply them to
 * their part of the global parameters, and send them back to the same clients.
 * There is one server shard per GPU. (Each GPU acts both as a client and as a
 * server shard.)
 */
void MultiNodeGraphGroup::setupServerShards() {
  calculateShardSizes();
  initShardGpuTensors();
  // CPU buffer for receiving/sending grads/params
  serverShardBufferCPU_ = std::vector<float>(nodeSizes_[mpi_my_rank_]);
  // Shard optimizers
  for(int shard = 0; shard < devices_.size(); shard++) {
    shardOptimizers_.push_back(Optimizer(options_));
  }
  // Mutexes to prevent simultaneous access to tensors and/or optimizers
  shardMutex_ = std::vector<std::mutex>(devices_.size());
}

/**
 * Calculate the size of each shard on this node.
 * Account for the edge case where the last shard has fewer parameters because
 * the node size is not perfectly divisibly by the number of shards.
 */
void MultiNodeGraphGroup::calculateShardSizes() {
  size_t nodeSize = nodeSizes_[mpi_my_rank_];
  size_t shardSize = ceilf(((float)nodeSize) / devices_.size());
  for(int shard = 0; shard < devices_.size(); shard++) {
    size_t remainingNodeSize = nodeSize - (shardSize * shard);
    // Takes care of edge case where last shard is smaller than the others
    shardSizes_.push_back(std::min(shardSize, remainingNodeSize));
  }
}

/**
 * Initialize the GPU tensors for storing the parameters and gradients of each
 * server shard.
 */
void MultiNodeGraphGroup::initShardGpuTensors() {
  size_t offset = 0;
  for(int i = 0; i < mpi_my_rank_; i++) {
    offset += nodeSizes_[i];
  }
  for(int shard = 0; shard < devices_.size(); shard++) {
    Tensor gpuParams
        = newTensor(shardSizes_[shard], clientGraphs_[shard]->getBackend());
    gpuParams->copyFrom(clientGraphs_[0]->params()->vals()->subtensor(
        offset, shardSizes_[shard]));
    shardParams_.push_back(gpuParams);
    shardGrads_.push_back(
        newTensor(shardSizes_[shard], clientGraphs_[shard]->getBackend()));
    offset += shardSizes_[shard];
  }
}

/**
 * Launch independent thread which continually receives gradients assigned to
 * this shard from any client, runs the shard optimizer and sends back the
 * updated parameters.
 */
void MultiNodeGraphGroup::launchServerThread() {
// @TODO: move CUDA stuff into separate .cu files and remove '&& CUDA_FOUND'
#if MPI_FOUND && CUDA_FOUND
  serverShardThread_ = new std::thread([this] {
    // keep track of number of nodes still communicating with this shard
    int nCommunicatingNodes = mpi_comm_world_size_;
    MPI_Status status;
    do {
      // Receive grads from any client
      unsigned long messageInfo[4];
      MPI_Recv(&messageInfo,
               4,
               MPI_UNSIGNED_LONG,
               MPI_ANY_SOURCE,
               MPI_TAG_GRAD_PUSH_MSG_,
               MPI_COMM_WORLD,
               &status);
      if(messageInfo[MSG_INFO_STATUS_] == STATUS_NODE_FINISHED_) {
        nCommunicatingNodes--;
        continue;
      }  // register finished node and skip to next loop iteration
      MPI_Recv(serverShardBufferCPU_.data(),
               nodeSizes_[mpi_my_rank_],
               MPI_FLOAT,
               status.MPI_SOURCE,
               MPI_TAG_GRAD_PUSH_,
               MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);

      // Update shard params asynchronously over GPUs
      std::vector<std::thread> threads;
      size_t offset = 0;
      for(int gpu = 0; gpu < devices_.size(); gpu++) {
        size_t size = shardSizes_[gpu];

        threads.emplace_back(std::thread(
            [=](int gpu, size_t offset, size_t size, size_t batchWords) {
              std::lock_guard<std::mutex> guard(shardMutex_[gpu]);

              // Copy grads to appropriate GPU
              cudaMemcpy(shardGrads_[gpu]->data(),
                         &serverShardBufferCPU_.at(offset),
                         size * sizeof(float),
                         cudaMemcpyHostToDevice);
              cudaStreamSynchronize(0);

              // Run optimizer on GPU
              if(scaleLearningRate_ && batchWords > 0) {
                shardOptimizers_[gpu]->update(shardParams_[gpu],
                                              shardGrads_[gpu],
                                              batchWords / avgBatchWords_);
              } else {
                shardOptimizers_[gpu]->update(shardParams_[gpu],
                                              shardGrads_[gpu]);
              }
              cudaStreamSynchronize(0);
              // Copy params from GPU
              cudaMemcpy(&serverShardBufferCPU_.at(offset),
                         shardParams_[gpu]->data(),
                         size * sizeof(float),
                         cudaMemcpyDeviceToHost);
              cudaStreamSynchronize(0);
            },
            gpu,
            offset,
            size,
            messageInfo[MSG_INFO_BATCHWORDS_]));

        offset += size;
      }
      for(auto &&t : threads) {
        t.join();
      }

      // Send updated params to same client
      MPI_Ssend(serverShardBufferCPU_.data(),
                nodeSizes_[mpi_my_rank_],
                MPI_FLOAT,
                status.MPI_SOURCE,
                MPI_TAG_PARAM_PUSH_,
                MPI_COMM_WORLD);

    } while(nCommunicatingNodes != 0);
  });
#endif
}

/**
 * Safely shut down the launched server shard thread.
 */
void MultiNodeGraphGroup::shutDownServerThread() {
  serverShardThread_->join();  // Wait for server thread to finish communicating
                               // (with unfinished nodes)
}

/**
 * Launch independent threads which continually synchronize their client's
 * gradients/parameters whenever the respective communication buffers are full.
 */
void MultiNodeGraphGroup::launchCommOverlapThreads() {
#if MPI_FOUND
  for(int gpu = 0; gpu < devices_.size(); gpu++) {
    clientCommThreads_.emplace_back(new std::thread(
        [this](int gpu) {
          do {
            // Wait for GPU (client) to fill buffers pointers
            std::unique_lock<std::mutex> uniqueLock(
                mutexClientCommOverlapBuffersFilled_[gpu]);
            while(!clientCommOverlapBuffersFilled_[gpu]) {
              cvClientCommOverlapBuffersFilled_[gpu].wait(uniqueLock);
            }

            if(stopClientCommThreads_) {
              break;
            }

            // Synchronize with server shards
            synchronizeWithServerShards(
                clientCommOverlapBuffersGPU_[gpu],
                clientCommOverlapBuffersGPU_[gpu],
                gpu,
                scaleLearningRate_ ? clientCommittedWordCounts_[gpu] : 0);

            // Indicate that buffers can be read from and filled again
            clientCommOverlapBuffersFilled_[gpu] = false;

          } while(!stopClientCommThreads_);
        },
        gpu));
  }
#endif
}

/**
 * Safely shut down the launched communication overlap threads
 */
void MultiNodeGraphGroup::shutDownCommOverlapThreads() {
  stopClientCommThreads_ = true;
  for(int gpu = 0; gpu < devices_.size(); gpu++) {
    clientCommOverlapBuffersFilled_[gpu] = true;
    cvClientCommOverlapBuffersFilled_[gpu]
        .notify_one();  // Unblock thread from lock, then join it
    clientCommThreads_[gpu]->join();
  }
}

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
void MultiNodeGraphGroup::synchronizeWithServerShards(Tensor newGrads,
                                                      Tensor oldParams,
                                                      int gpu,
                                                      size_t batchWords) {
// @TODO: move CUDA stuff into separate .cu files and remove '&& CUDA_FOUND'
#if MPI_FOUND && CUDA_FOUND
  size_t offset = 0;
  for(int node = 0; node < mpi_comm_world_size_; node++) {
    size_t nodeSize = nodeSizes_[node];

    // Update remotely if node != this node
    if(node != mpi_my_rank_) {
      Tensor gradient;

      // Delayed Gradient Update
      if(tau_ > 1) {
        std::lock_guard<std::mutex> guard(optDelayMutex_[node]);
        accGradientBuffer[node]->copyFrom(
            newGrads->subtensor(offset, nodeSize));
        // Accumulate the gradient
        using namespace functional;
        Element(_1 += _2, accGradients[node], accGradientBuffer[node]);
        // Accumulate total batch word
        totalBatchWords[node] += batchWords;
        delay_count[node]++;

        if(delay_count[node] < tau_)
          continue;
        delay_count[node] = 0;
        gradient = accGradients[node];
        batchWords = totalBatchWords[node];
      } else {
        gradient = newGrads->subtensor(offset, nodeSize);
      }

      // Copy grads from GPU to CPU (for MPI sending)
      cudaMemcpy(clientCommBuffersCPU_[gpu].data(),
                 gradient->data(),
                 nodeSize * sizeof(float),
                 cudaMemcpyDeviceToHost);
      cudaStreamSynchronize(0);

      // Send grads to server node
      size_t messageInfo[4];
      messageInfo[MSG_INFO_SIZE_] = nodeSize;
      messageInfo[MSG_INFO_CLIENT_] = gpu;
      messageInfo[MSG_INFO_BATCHWORDS_] = batchWords;
      messageInfo[MSG_INFO_STATUS_] = STATUS_NODE_TRAINING_;
      MPI_Ssend(&messageInfo,
                4,
                MPI_UNSIGNED_LONG,
                node,
                MPI_TAG_GRAD_PUSH_MSG_,
                MPI_COMM_WORLD);
      MPI_Ssend(clientCommBuffersCPU_[gpu].data(),
                nodeSize,
                MPI_FLOAT,
                node,
                MPI_TAG_GRAD_PUSH_,
                MPI_COMM_WORLD);
      // Reset total gradient and batch words
      if(tau_ > 1) {
        std::lock_guard<std::mutex> guard(optDelayMutex_[node]);
        accGradients[node]->set(0);
        totalBatchWords[node] = 0;
      }
      // Receive updated params from server node
      MPI_Recv(clientCommBuffersCPU_[gpu].data(),
               nodeSize,
               MPI_FLOAT,
               node,
               MPI_TAG_PARAM_PUSH_,
               MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);

      // Copy params from CPU back to GPU
      cudaMemcpy(oldParams->subtensor(offset, nodeSize)->data(),
                 clientCommBuffersCPU_[gpu].data(),
                 nodeSize * sizeof(float),
                 cudaMemcpyHostToDevice);
      cudaStreamSynchronize(0);

      // Else update locally if node == this node
    } else {
      size_t localOffset = offset;
      std::vector<std::thread> threads;

      for(int gpu = 0; gpu < devices_.size(); gpu++) {
        size_t gpuSize = shardSizes_[gpu];

        threads.emplace_back(std::thread(
            [=](int gpu, size_t offset, size_t size) {
              std::lock_guard<std::mutex> guard(shardMutex_[gpu]);

              // Copy grads to appropriate GPU
              shardGrads_[gpu]->copyFrom(newGrads->subtensor(offset, size));
              // Run optimizer on GPU
              if(scaleLearningRate_ && batchWords > 0) {
                shardOptimizers_[gpu]->update(shardParams_[gpu],
                                              shardGrads_[gpu],
                                              batchWords / avgBatchWords_);
              } else {
                shardOptimizers_[gpu]->update(shardParams_[gpu],
                                              shardGrads_[gpu]);
              }
              cudaStreamSynchronize(0);
              // Copy params back to current GPU
              oldParams->subtensor(offset, size)->copyFrom(shardParams_[gpu]);
            },
            gpu,
            localOffset,
            gpuSize));

        localOffset += gpuSize;
      }
      for(auto &&t : threads) {
        t.join();
      }
    }

    offset += nodeSize;
  }
#endif
}

/**
 * Execute given batch on this node, pushing/pulling the resulting
 * gradients/parameters to/from the server shards
 * or -- if comm. overlap enabled -- to/from the communication buffers, summing
 * gradients locally if the communication thread is busy
 *
 * @param batch Batch on which to perform forward and backward passes.
 */
void MultiNodeGraphGroup::execute(Ptr<data::Batch> batch) {
  if(!initialized_) {
    init(batch);
    initialized_ = true;
  }

  auto task = [this](Ptr<data::Batch> batch) {
    static size_t i = 0;
    thread_local Ptr<ExpressionGraph> graph;
    thread_local Ptr<models::ModelBase> builder;
    thread_local size_t my_id = 0;
    thread_local size_t t = 0;
    // only for scheduler statistic
    thread_local float cost = 0;
    thread_local size_t num_seen_words = 0;
    thread_local size_t num_seen_sentences = 0;

    if(!graph) {
      std::lock_guard<std::mutex> lock(mutexClientInit_);
      my_id = i;
      graph = clientGraphs_[i];
      builder = clientBuilders_[i++];
    }

    auto costNode = builder->build(graph, batch);

#if MPI_FOUND
    if(t == 0) {
      MPI_Barrier(MPI_COMM_WORLD);
      if(my_id != 0)
        graph->params()->vals()->copyFrom(clientGraphs_[0]->params()->vals());
      MPI_Barrier(MPI_COMM_WORLD);
    }
#endif

    graph->forward();
    cost += costNode->scalar();
    num_seen_words += batch->words();
    num_seen_sentences += batch->size();
    graph->backward();

    t++;

    graph->getBackend()->synchronize();

    if(!clientCommOverlap) {
      synchronizeWithServerShards(graph->params()->grads(),
                                  graph->params()->vals(),
                                  my_id,
                                  batch->wordsTrg());
    }

    // Overlapping computations with communication
    if(clientCommOverlap) {
      // Add computed gradients to local running sum
      Element(functional::_1 = functional::_1 + functional::_2,
              clientSummedGradsGPU[my_id],
              graph->params()->grads());
      graph->getBackend()->synchronize();

      // Sum up word counts if batch flexible learning rate is enabled
      if(scaleLearningRate_) {
        clientSummedWordCounts_[my_id] += batch->wordsTrg();
      }

      // If communication channel ready, swap graph's pointers with secondary
      // buffers
      if(!clientCommOverlapBuffersFilled_[my_id]) {
        std::unique_lock<std::mutex> tryLock(
            mutexClientCommOverlapBuffersFilled_[my_id], std::try_to_lock);
        if(tryLock.owns_lock()) {
          // Copy parameters from communication buffer
          graph->params()->vals()->copyFrom(
              clientCommOverlapBuffersGPU_[my_id]);
          // Copy summed grads to communication buffer
          clientCommOverlapBuffersGPU_[my_id]->copyFrom(
              clientSummedGradsGPU[my_id]);
          // Commit summed word counts if batch-flexible-lr enabled
          if(scaleLearningRate_) {
            clientCommittedWordCounts_[my_id] = clientSummedWordCounts_[my_id];
            clientSummedWordCounts_[my_id] = 0;
          }
          // Notify communication thread that buffers have been read and filled
          clientCommOverlapBuffersFilled_[my_id] = true;
          cvClientCommOverlapBuffersFilled_[my_id].notify_one();
          // Apply summed gradients to new parameters
          clientLocalOptimizers_[my_id]->update(graph->params()->vals(),
                                                clientSummedGradsGPU[my_id]);
          // Clear summed gradients
          clientSummedGradsGPU[my_id]->set(0);
        }
      }
    }

    // Run scheduler (if enabled)
    if(t % tau_ == 0 && scheduler_) {
      std::unique_lock<std::mutex> lock(schedulerMutex_);

      // Wait until the thread that wants to do validation is finished.
      clientThreadPool_->wait_for_one(lock);

      if(options_->get<std::string>("cost-type") != "ce-sum")
        cost /= tau_;

      if(tau_ > 1) {
        std::vector<size_t> fakeLength = {1, 1};
        auto fb = data::CorpusBatch::fakeBatch(
            fakeLength, num_seen_sentences, NULL);
        fb->front()->setWords(num_seen_words);
        scheduler_->update(cost, fb);
      } else {
        scheduler_->update(cost, batch);
      }

      num_seen_words = 0;
      num_seen_sentences = 0;
      cost = 0;

      if((scheduler_->saving() || scheduler_->validating())) {
        // Wait with validation or saving until all other threads are done with
        // update.
        // We want to reuse the graphs for validation, so they need to be in
        // a safe state.
        clientThreadPool_->wait_for_others(lock);
#if MPI_FOUND
        // wait until other nodes are ready
        MPI_Barrier(MPI_COMM_WORLD);

        // TODO: Saving is broken
        // if(mpi_my_rank_ == 0 && scheduler_->saving())
        //  this->save(graph);

        if(mpi_my_rank_ == 0 && scheduler_->validating())
          scheduler_->validate(clientGraphs_);

        // inform other nodes to continue
        MPI_Barrier(MPI_COMM_WORLD);
#endif
        // Validation or saving is done, tell other threads to continue work.
        clientThreadPool_->notify_others();
      }
    }
  };

  clientThreadPool_->enqueue(task, batch);
}

/**
 * Notify server shards that this node has finished training.
 */
void MultiNodeGraphGroup::signalFinishedToServerShards() {
#if MPI_FOUND
  unsigned long messageInfo[4];
  messageInfo[MSG_INFO_STATUS_] = STATUS_NODE_FINISHED_;
  for(int node = 0; node < mpi_comm_world_size_; node++) {
    MPI_Ssend(&messageInfo,
              4,
              MPI_UNSIGNED_LONG,
              node,
              MPI_TAG_GRAD_PUSH_,
              MPI_COMM_WORLD);
  }
#endif
}
}
