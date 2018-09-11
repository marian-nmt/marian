#include "training/graph_group_multinode_sync.h"
#include "functional/functional.h"
#include "tensors/tensor_operators.h"

namespace marian {

void MultiNodeGraphGroupSync::updateAvgParams(Tensor paramsAvg,
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
  // @TODO: Is this specific to multi-node?
  scheduler_->registerTrainingObserver(scheduler_);

  scheduler_->registerTrainingObserver(syncOptimizer_);
}

/**
 * Allocate new tensor on given GPU and store allocator.
  // @TODO: Is this specific to multi-node?
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
  int network_size = (int)clientGraphs_[0]->params()->vals()->size();
  LOG(info, "model size = {} float params", network_size);

  // @TODO: move this after the other allocations, unless there is a reason
  // @TODO: should this code know how to allocate this? Shouldn't this be owned by the parameter-averager?
  if(movingAvg_)
    paramsAvg_ = newTensor(network_size, clientGraphs_[0]->getBackend());
  // @TODO: original version put averaging onto a different GPU, to save GPU RAM.
  //        No longer needed since we no longer aggregate anything else on the GPU.

  // setup sync sgd storage, We keep the summed gradient on device 0
  // @TODO: eliminate devices size condition once comm_ works
  //if (devices_.size() > 1)
  //  sumGradientBuffer_ = newTensor(network_size, clientGraphs_[0]->getBackend());
  if (tau_ > 1)
    accGradient_       = newTensor(network_size, clientGraphs_[0]->getBackend());
}

/**
 * Initialize the CPU arrays that are used during transferring data via MPI.
 * This uses pinned memory for faster CudaMemCpy operations. Requires the graph
 * to be initialized first so we know its size.
 * @TODO: these buffers are fully owned by one function, we should only touch them there for improved code locality
 */
void MultiNodeGraphGroupSync::initCPUArrays() {
  //accGradientsSync_cpu
  //    = std::vector<float>(clientGraphs_[0]->params()->vals()->size());
  //receiveBuffer_cpu
  //    = std::vector<float>(clientGraphs_[0]->params()->vals()->size());
}

/**
 * Setup clients that will compute gradients and communicate them with the
 * server shards.
 * There is one client per GPU.
 */
void MultiNodeGraphGroupSync::setupClients(Ptr<data::Batch> batch) {
  runBatchThroughClientGraphs(batch);   // @TODO: what is this? A fake batch?
  initCPUArrays();
}

/**
 * Initialize the graphs (models) of all clients on this node with the given
 * batch.
 * @TODO: why is his a separate function? Is this used more than once?
 */
void MultiNodeGraphGroupSync::runBatchThroughClientGraphs(
    Ptr<data::Batch> batch) {
  for(int i = 0; i < devices_.size(); i++) {
    THREAD_GUARD(clientBuilders_[i]->build(clientGraphs_[i], batch);
                 clientGraphs_[i]->forward();
                 clientGraphs_[i]->getBackend()->synchronize(););
  }
}

/**
 * Adds one GPU's gradient tensor to accGradient_
 * sumGradientBuffer_ has been allocated on the same GPU as accGradient_.
 * @TODO: inline this to where it is called, to make the synchronization situation clearer
 */
void MultiNodeGraphGroupSync::sumGRAD(Tensor gradient) { // @TODO: why UPPERCASE?
  ABORT_IF(!(devices_.size() > 1 || tau_ > 1), "unnecessarily summing gradient??");
  ABORT_IF(accGradient_ == nullptr, "accGradient not created??");

  // wait for GPU computation to complete
  // The first GPU that finishes will get to call sumGRAD() first, and therefore gets sumGradientBuffer first.
  if (accGradient_->getDeviceId() != gradient->getDeviceId())
    gradient->getBackend()->synchronize();

  std::lock_guard<std::mutex> guard(sumGradientMutex_);
  if (accGradient_->getDeviceId() != gradient->getDeviceId()) { // don't copy if already on the same device as accumulator
    ABORT_IF(sumGradientBuffer_ == nullptr, "no sumGradientBuffer_ when we need it??");
    ABORT_IF(sumGradientBuffer_->getDeviceId() != accGradient_->getDeviceId(), "sumGradientBuffer_ on wrong device??");
    sumGradientBuffer_->copyFrom(gradient);
    gradient = sumGradientBuffer_; // substitute gradient reference with the a ref to the copy on the right device
  }
  else
    LOG(info, "not copying gradient on dev {}", gradient->getDeviceId().no);

  using namespace functional;
  Element(_1 += _2, accGradient_, gradient);
  // Note: This merely submits the GPU compute on device 0, but does not wait for its completion.
  // The next copyFrom() call will run on the same stream, and thus after this is complete.
  // The lock_guard will only guard CPU-side aspects of accessing sumGradientBuffer.
}

/**
 * All-reduce accGradientSync across nodes.
 */
// @TODO: This function mixes several concerns. It should be split up into:
//  - all-reduce of gradient across workers -> allReduceAccGradients()
//  - model update (should just be in execute())
//  - broadcast of updated model to all devices (this mirrors sumGRAD())
// @TODO: this goes away in case of NCCL; or rather, this should be moved to DefaultCommunicator
void MultiNodeGraphGroupSync::sendReceiveUpdateSync(Tensor accGradientsSync) {
  auto network_size = clientGraphs_[0]->params()->vals()->size();

  // Copy the locally aggregated gradients to the CPU
  accGradientsSync->get(/*out*/ accGradientsSync_cpu);

  // Wait until all nodes are ready
  // @TODO: Is that necessary before allReduce?
  //mpi_->barrier();

  //LOG(info, "all-reducing {} gradient values, node {}", network_size, mpi_->myRank());
  receiveBuffer_cpu.resize(network_size);
  mpi_->allReduce(accGradientsSync_cpu.data(),  // CPU buffers
                  receiveBuffer_cpu.data(),
                  network_size,
                  MPI_FLOAT, MPI_SUM);

  // Copy the data back to the GPU
  clientGraphs_[0]->params()->grads()->set(receiveBuffer_cpu);
}

/**
 * second step of update  --@TODO: split this up correctly; e.g. move back to execute()
 * We are back in GPU land at this point.
 */
void MultiNodeGraphGroupSync::sendReceiveUpdateSync2() {
  // Perform optimizer step
  syncOptimizer_->update(clientGraphs_[0]);

  if(movingAvg_)
    updateAvgParams(paramsAvg_,
                    clientGraphs_[0]->params()->vals(),
                    scheduler_->numberOfBatches());

  // Distribute the updated params to the rest of the devices
  // @TODO: Does the multi-threaded operation here add any value for GPUs?
  std::vector<std::thread> threads; // @TODO: keep the thread pool around
  for(int idx = 1; idx < devices_.size(); idx++) {
    threads.emplace_back(std::thread(
        [=]() {
          clientGraphs_[idx]->params()->vals()->copyFrom(
              clientGraphs_[0]->params()->vals());
        }));
  }
  for(auto&& t : threads) {
    t.join();
  }
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

  //for (size_t i = 0; i < devices_.size(); i++) {
  //  const auto& b = batches[i];
  //  LOG(info, "[rank {}] [dev {} {}] max len {} total {} (source)", mpi_->myRank(), i, devices_[i], b->width(), b->words());
  //}

  static int t = 0;

  static float cost = 0;
  static size_t num_seen_words = 0;
  static size_t num_seen_sentences = 0;

  // compute gradients
  auto task = [&](int my_id) {
    auto batch = batches[my_id];
    auto graph = clientGraphs_[my_id];
    auto builder = clientBuilders_[my_id];

    auto costNode = builder->build(graph, batch);

    if(t == 0) { // the very first update must start from identical initial values
      if(my_id != 0)
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

    //if (accGradient_ != nullptr) // aggregate locally across the devices
    //  sumGRAD(graph->params()->grads());
    //else // if only one GPU and not summing, we directly work off the gradient to save GPU RAM
    //  ABORT_IF(devices_.size() > 1 || tau_ > 1, "no accGradient_ when we need it??");
  };

  if (devices_.size() > 1) {
    ThreadPool pool(devices_.size(), devices_.size()); // @TODO: use comm_->foreach()
    for(int idx = 0; idx < devices_.size(); ++idx)
      pool.enqueue(task, idx);
    // destruction of pool joins the threads
  }
  else { // don't use thread if single GPU
    task(0);
  }

#if 1 // NCCL version
  ABORT_IF(accGradient_ != nullptr, "tau_ not yet implemented for NCCL MPI verseion"); // @TODO: we better change backward() to allow not resetting
  if (t % tau_ == 0) {
    //LOG(info, "NCCL/MPI reduce");
    // this is tricky because we want a "some-reduce" that computes the sum of all but redistributes only to the first device of each node
    if (commWithinNode_)
      commWithinNode_->reduceGrads();     // reduce local devices -> dev 0 contains node-local gradient
    if (commAcrossNodes_)
      commAcrossNodes_->allReduceGrads(); // all-reduce all dev 0 across nodes -> dev 0 of all devices contain the cross-node gradient
    sendReceiveUpdateSync2();
  }
#else
  // aggregate locally
  if (commWithinNode_)
    commWithinNode_->reduceGrads();

  // aggregate across delayed batches
  // @TODO: If we instead not reset the gradients themselves, we can eliminate accGradient_ altogether & save 0.5 GB.
  if (accGradient_ != nullptr) // aggregate locally across the devices
    sumGRAD(clientGraphs_[0]->params()->grads());

  // aggregate globally
  if (t % tau_ == 0) {
    sendReceiveUpdateSync(accGradient_ != nullptr ? accGradient_ : clientGraphs_[0]->params()->grads());
    sendReceiveUpdateSync2();

    // set the accumulating buffers to zero;
    if (accGradient_ != nullptr) // aggregate locally across the devices
      accGradient_->set(0); // @TODO: we can already launch this once we have copied it out, so that this op runs concurrently with the MPI exchange
  }
#endif

  t++;

  // Run scheduler (if enabled)
  if(t % tau_ == 0 && scheduler_) {
    // some objectives work on averages
    // @TODO: we need the precise word count for ce-sum
    if(options_->get<std::string>("cost-type") != "ce-sum")
      cost /= (tau_ * devices_.size());

    if(tau_ > 1) {
      // run a fake update
      std::vector<size_t> fakeLength = {1, 1};
      auto fb
          = data::CorpusBatch::fakeBatch(fakeLength, num_seen_sentences, NULL);
      fb->front()->setWords(num_seen_words);
      scheduler_->update(cost, fb);
    } else {
      // real update
      scheduler_->update(cost, fullBatch);
    }

    num_seen_words = 0;
    num_seen_sentences = 0;
    cost = 0;

    if((scheduler_->saving() || scheduler_->validating())) {
      // wait until other nodes are ready
      mpi_->barrier();

      // save
      // TODO: Saving is broken
      if(scheduler_->saving())
        save(/*graph*/);

      // validate
      if(mpi_->myRank() == 0 && scheduler_->validating()) {
        // temporarily save current params
        if (movingAvg_)
          clientGraphs_[0]->params()->vals()->get(accGradientsSync_cpu);
          //accGradient_->copyFrom(clientGraphs_[0]->params()->vals()); // we don't need to occupy GPU RAM for this

        if(movingAvg_)
          for(auto graph : clientGraphs_)
            graph->params()->vals()->copyFrom(paramsAvg_); // @TODO: any way to just swap the two?

        scheduler_->validate(clientGraphs_);

        if(movingAvg_)
          for(auto graph : clientGraphs_)
            graph->params()->vals()->set(accGradientsSync_cpu);
            //graph->params()->vals()->copyFrom(accGradient_);
      }

      // inform other nodes to continue
      mpi_->barrier();
    }
  }
}
}  // namespace marian
