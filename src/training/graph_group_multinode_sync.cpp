#include "training/graph_group_multinode_sync.h"
#include "functional/functional.h"
#include "tensors/tensor_operators.h"

namespace marian {

void MultiNodeGraphGroupSync::updateMovingAverage(Tensor paramsAvg,
                                         Tensor params,
                                         size_t batches) {
  using namespace functional;
  float decay
      = std::max(mvDecay_, 1.f - (float)(batches + 1) / (float)(batches + 10));
  Element(_1 = ((1.f - decay) * _1) + (decay * _2), paramsAvg, params);
}

/**
 * Set given scheduler to register training observers on the shard optimizers.
 */
void MultiNodeGraphGroupSync::setScheduler(Ptr<Scheduler> scheduler) {
  scheduler_ = scheduler;
  // optimizer has to be registered last to see a change of learning rate
  scheduler_->registerTrainingObserver(scheduler_);

  scheduler_->registerTrainingObserver(syncOptimizer_);

}

/**
 * Allocate new tensor on given GPU and store allocator.
 */
Tensor MultiNodeGraphGroupSync::newTensor(int size, Ptr<Backend> backend) {
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
void MultiNodeGraphGroupSync::init(Ptr<data::Batch> batch) {
  // Setup clients and shards
  setupClients(batch);
  int network_size = clientGraphs_[0]->params()->vals()->size();
  LOG(info, "model size = {} float params" , network_size);
  if (movingAvg_)
    paramsAvg_ = newTensor(network_size, clientGraphs_.back()->getBackend());

  // setup sync sgd storage, We keep the summed gradient on Node 0
  sumGradientBuffer = newTensor(network_size, clientGraphs_[0]->getBackend());
  accGradientsSync = newTensor(network_size, clientGraphs_[0]->getBackend());
}


/**
 * Initialize the CPU arrays, with pinned memory for faster CudaMemCpy operations.
 * Requires the graph to be initialized first so we know its size
 */
void MultiNodeGraphGroupSync::initCPUArrays() {
  accGradientsSync_cpu = std::vector<float>(clientGraphs_[0]->params()->vals()->size());
  receiveBuffer_cpu = std::vector<float>(clientGraphs_[0]->params()->vals()->size());
}

/**
 * Setup MPI world size and rank of this node.
 */
void MultiNodeGraphGroupSync::setupMPI() {
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
void MultiNodeGraphGroupSync::setupClients(Ptr<data::Batch> batch) {
  runBatchThroughClientGraphs(batch);
  initCPUArrays();

  clientThreadPool_ = new marian::ThreadPool(devices_.size(), devices_.size());
}

/**
 * Initialize the graphs (models) of all clients on this node with the given
 * batch.
 */
void MultiNodeGraphGroupSync::runBatchThroughClientGraphs(Ptr<data::Batch> batch) {
  for(int i = 0; i < devices_.size(); i++) {
    THREAD_GUARD(clientBuilders_[i]->build(clientGraphs_[i], batch);
                 clientGraphs_[i]->forward();
                 clientGraphs_[i]->getBackend()->synchronize(););
  }
}

/**
 * Initialize variables required for overlapping client computations and
 * communication.
 * Includes summed and committed word counts, buffer flags, mutexes and
 * condition variables.
 */
void MultiNodeGraphGroupSync::sumGRAD(Tensor gradient) {
  std::lock_guard<std::mutex> guard(sumGradientMutex_);
  sumGradientBuffer->copyFrom(gradient);
  using namespace functional; //@TODO makes more sense to do that on the CPU i think
  Element(_1 += _2, accGradientsSync, sumGradientBuffer);
}

/**
 * If it's rank 0, it's a local update, if it's rank one it's remote
 * send and receive. Make sure you only call from device 0.
 */
void MultiNodeGraphGroupSync::sendReceiveUpdateSync() {
  #if MPI_FOUND
  int network_size = accGradientsSync_cpu.size();

  // Copy the data to the CPU
  accGradientsSync->get(accGradientsSync_cpu);

  // Wait until all nodes are ready
  MPI_Barrier(MPI_COMM_WORLD);

  int reduce_result = MPI_Allreduce(accGradientsSync_cpu.data(), //CPU buffers
              receiveBuffer_cpu.data(),
              network_size,
              MPI_FLOAT,
              MPI_SUM,
              MPI_COMM_WORLD);

  // Copy the data back to the GPU and do optimizer update
  // Do update with last GPU to distribute the memory
  clientGraphs_.back()->params()->grads()->set(receiveBuffer_cpu);

  // Perform optimizer step
  syncOptimizer_->update(clientGraphs_.back());

  if(movingAvg_)
      updateMovingAverage(
        paramsAvg_, clientGraphs_.back()->params()->vals(),
        scheduler_->numberOfBatches());

  //Distribute the graph to the rest of the devices
  std::vector<std::thread> threads;
  for(int idx = 0; idx < devices_.size() - 1; idx++) {
    threads.emplace_back(std::thread(
        [=](int idx) {
          clientGraphs_[idx]->params()->vals()->copyFrom(
            clientGraphs_.back()->params()->vals());
        },
        idx));
  }
  for(auto&& t : threads) {
    t.join();
  }

  //set the accumulating buffers to zero;
  accGradientsSync->set(0);
  std::fill(accGradientsSync_cpu.begin(), accGradientsSync_cpu.end(), 0);
  std::fill(receiveBuffer_cpu.begin(), receiveBuffer_cpu.end(), 0);
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
void MultiNodeGraphGroupSync::execute(Ptr<data::Batch> fullBatch) {
  if(!initialized_) {
    init(fullBatch);
    initialized_ = true;
  }

  std::vector<Ptr<data::Batch>> batches = fullBatch->split(devices_.size());

  static int t = 0;

  static float cost = 0;
  static size_t num_seen_words = 0;
  static size_t num_seen_sentences = 0;

  {
    auto task = [this, batches](int my_id) {
      auto batch = batches[my_id];
      auto graph = clientGraphs_[my_id];
      auto builder = clientBuilders_[my_id];

      auto costNode = builder->build(graph, batch);

      if (t == 0) {
        if (my_id != 0)
          graph->params()->vals()->copyFrom(clientGraphs_[0]->params()->vals());
      }

      graph->forward();
      {
        std::lock_guard<std::mutex> guard(sumCostMutex_);
        cost += costNode->scalar();
        num_seen_words += batch->words();
        num_seen_sentences += batch->size();
      }
      graph->backward();

      graph->getBackend()->synchronize(); //@Alham do you know why we need this here?

      sumGRAD(graph->params()->grads());
    };

    ThreadPool pool(devices_.size(), devices_.size());
    for(int idx = 0; idx < devices_.size(); ++idx)
      pool.enqueue(task, idx);
  }

  if (t % tau_ == 0)
    sendReceiveUpdateSync();    
  
  t++; 

  // Run scheduler (if enabled)
  if(t % tau_ == 0 && scheduler_) {
    if (options_->get<std::string>("cost-type") != "ce-sum")
      cost /= (tau_ * devices_.size());

    if (tau_ > 1) {
      std::vector<size_t> fakeLength = {1, 1};
      auto fb = data::CorpusBatch::fakeBatch(fakeLength,
                                        num_seen_sentences,
                                        NULL);
      fb->front()->setWords(num_seen_words);
      scheduler_->update(cost, fb);
    } else {
      scheduler_->update(cost, fullBatch);
    }
    
    num_seen_words = 0;
    num_seen_sentences = 0;
    cost = 0;

    if((scheduler_->saving() || scheduler_->validating())) {
      #if MPI_FOUND
      //wait until other nodes are ready
      MPI_Barrier(MPI_COMM_WORLD);

      // TODO: Saving is broken
      //if(mpi_my_rank_ == 0 && scheduler_->saving())
      //  this->save(graph);

      if(mpi_my_rank_ == 0 && scheduler_->validating()) {
        // temporarily save current params
        if(movingAvg_)
          accGradientsSync->copyFrom(clientGraphs_[0]->params()->vals());

        if(movingAvg_)
          for(auto graph : clientGraphs_)
            graph->params()->vals()->copyFrom(paramsAvg_);

        scheduler_->validate(clientGraphs_);

        if(movingAvg_)
          for(auto graph : clientGraphs_)
            graph->params()->vals()->copyFrom(accGradientsSync);
      }

      // inform other nodes to continue
      MPI_Barrier(MPI_COMM_WORLD);
      #endif
    }
  }

}
}
