// @TODO: rename to communicator_nccl.h
// Note: This must only be included if defined(CUDA_FOUND) && defined(USE_NCCL)
// clang-format off
#include "training/communicator.h"
//#include "functional/functional.h"
//#include "tensors/tensor_operators.h"
// clang-format on
 
#include "cuda_runtime.h"
#include "nccl.h"
#include "tensors/gpu/cuda_helpers.h"

namespace marian {

class NCCLCommunicator : public ICommunicator {
private:
  std::vector<ncclComm_t> comms_;     // [device index]
  std::vector<cudaStream_t> streams_; // [device index]
  std::vector<int> devices_;          // [device index]
  Ptr<IMPIWrapper> mpi_; // non-null if multi-node

  static void groupStart() { NCCLCHECK(ncclGroupStart()); } // helpers to make sure we check the error
  static void groupEnd()   { NCCLCHECK(ncclGroupEnd());   }

  void synchronizeAll() {
    for(int i = 0; i < graphs_.size(); ++i) {
      CUDA_CHECK(cudaSetDevice(devices_[i]));
      CUDA_CHECK(cudaStreamSynchronize(streams_[i]));
    }
  }

  size_t myRank(size_t localDeviceIndex) const { // map local device index to a global rank
    if (mpi_)
      return mpi_->myRank() * devices_.size() + localDeviceIndex;
    else
      return localDeviceIndex;
  }

  size_t numWorkers() const { // total number of devices across all workers
    if (mpi_)
      return mpi_->commWorldSize() * devices_.size();
    else
      return devices_.size();
  }

  size_t dataSize() const { // total number of floats that comprise the concatenated parameter and gradient vector
    return graphs_[0]->params()->vals()->size();
  }

  // determine the (max) shard size
  // All shards except the last one have this size.
  // Presently, even all shards must have identical size, due to a limitation in NCCL we have not yet worked around.
  size_t shardSize() const {
    size_t numShards = numWorkers();
    size_t size = (dataSize() + numShards - 1) / numShards;
#if 1 // for now, all shards must have the same size, since NCCL does not allow a sub-slice for the last shard
    ABORT_IF(size * numShards != dataSize(), "presently, all shards must have the same size");
#endif
    return size;
  }

  // determine the index range (begin, end) of a shard
  std::pair<size_t, size_t> shardRange(size_t localDeviceIndex) const {
    size_t size = shardSize();
    size_t rank = myRank(localDeviceIndex);
    size_t begin = rank * size;
    size_t end = begin + size;
    end = std::min(end, dataSize()); // clip last shard. Note: Presently this never happens, since shardSize() enforces uniform shard size.
    return{ begin, end };
  }

public:
  // a NCCLCommunicator is bound to a set of graphs, one per GPU device
  // If MPI is used, then each worker has an instance of this class for its specific
  // set of GPU devices, which are communicating with each other.
  NCCLCommunicator(const std::vector<Ptr<ExpressionGraph>>& graphs, Ptr<IMPIWrapper> mpi)
      : ICommunicator(graphs),
        comms_(graphs.size()),
        streams_(graphs.size()),
        devices_(graphs.size()),
        mpi_(mpi) {
    if (mpi_)
      LOG(info, "[comm] Using NCCL library and MPI for GPU communication");
    else
      LOG(info, "[comm] Using NCCL library for GPU communication");

    for(int i = 0; i < graphs_.size(); ++i) {
      auto device = graphs_[i]->getBackend()->getDeviceId();

      ABORT_IF(device.type != DeviceType::gpu,
               "NCCL communicator can only be used with GPUs");

      devices_[i] = device.no;
      CUDA_CHECK(cudaSetDevice(devices_[i]));
      CUDA_CHECK(cudaStreamCreate(&streams_[i]));
    }

    // when using MPI, the setup is a laborious
    // cf. https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/index.html#multidevprothrd
    if (mpi_) {
      // generate NCCL unique ID at one process and broadcast to all
      ncclUniqueId uniqueId = { 0 };
      LOG(info, "[{}] before ncclGetUniqueId", mpi_->to_string());
      if (mpi->myRank() == 0)
        NCCLCHECK(ncclGetUniqueId(&uniqueId));
      LOG(info, "[{}] before bcast", mpi_->to_string());
      //LOG(info, "before bcast: unique id = {}", std::string(uniqueId.internal, NCCL_UNIQUE_ID_BYTES));
      static_assert(sizeof(uniqueId) == NCCL_UNIQUE_ID_BYTES, "wrong NCCL_UNIQUE_ID_BYTES??"); // (this value is used in NVidia examples)
      mpi_->bCast(&uniqueId, sizeof(uniqueId), MPI_BYTE, 0);
      LOG(info, "[{}] after bcast", mpi_->to_string());
      //LOG(info, "unique id = {}", std::string(uniqueId.internal, NCCL_UNIQUE_ID_BYTES));

      // if more than one device then initialize NCCL with group API
      //if (devices_.size() > 1) {
        groupStart();
        for (int localDeviceIndex = 0; localDeviceIndex < devices_.size(); localDeviceIndex++) {
          CUDA_CHECK(cudaSetDevice(devices_[localDeviceIndex]));
          LOG(info, "[{}] ncclCommInitRank {} out of {}, GPU[{}]", mpi_->to_string(), myRank(localDeviceIndex), numWorkers(), localDeviceIndex);
          NCCLCHECK(ncclCommInitRank(&comms_[localDeviceIndex], numWorkers(), uniqueId, myRank(localDeviceIndex)));
          LOG(info, "[{}] done ncclCommInitRank {} out of {}, GPU[{}]", mpi_->to_string(), myRank(localDeviceIndex), numWorkers(), localDeviceIndex);
        }
        groupEnd();
        LOG(info, "[{}] group done constructing NCCLCommunicator", mpi_->to_string());
      //}
      //// one device: no group API
      //else {
      //  CUDA_CHECK(cudaSetDevice(devices_[0]));
      //  LOG(info, "[mpi rank {} of {}] ncclCommInitRank", mpi_->myRank(), mpi_->commWorldSize());
      //  NCCLCHECK(ncclCommInitRank(&comms_[0], mpi_->commWorldSize(), uniqueId, mpi_->myRank()));
      //  LOG(info, "[mpi rank {}] done constructing NCCLCommunicator", mpi_->myRank());
      //}
    }
    // without MPI, we have a handy convenience version to initialize
    // @TODO: We should be able to just use the code above as well.
    else {
      LOG(info, "ncclCommInitAll");
      NCCLCHECK(ncclCommInitAll(comms_.data(), devices_.size(), devices_.data()));
      LOG(info, "done ncclCommInitAll");
      LOG(info, "done constructing NCCLCommunicator");
    }
  }

  ~NCCLCommunicator() override {
    for(int i = 0; i < devices_.size(); ++i) {
      cudaSetDevice(devices_[i]);
      cudaStreamDestroy(streams_[i]);
      ncclCommDestroy(comms_[i]);
    }
  }

  void foreach(const std::function<void(size_t, size_t /*shardBegin*/, size_t /*shardEnd*/)>& func, bool parallel= true) const override {
    parallel &= graphs_.size() > 1;
      
    std::vector<std::thread> group;
    // iterate over all shards on this worker
    size_t begin, end;
    for(size_t i = 0; i < graphs_.size(); ++i) {
      std::tie
      (begin, end) = shardRange(i);
      //std::cerr << "[" << mpi_->to_string() << "] foreach " << begin << " " << end << std::endl;
      size_t size = end-begin;

      if (parallel)
        group.emplace_back(func, i, begin, end);
      else
        func(i, begin, end);
    }
    for(auto& t : group) // (note: group is empty is not parallel)
      t.join();
  }

  void scatterReduce() override {
    size_t begin, end;
    groupStart();
    for(int i = 0; i < graphs_.size(); ++i) {
      std::tie
      (begin, end) = shardRange(i);
      //std::cerr << "[" << mpi_->to_string() << "] scatterReduce " << begin << " " << end << std::endl;

      auto grads = graphs_[i]->params()->grads();
      const auto* sendbuf = grads->data();
      auto*       recvbuf = grads->subtensor(begin, end-begin)->data();
      size_t      bufsize = shardSize();

      NCCLCHECK(ncclReduceScatter(sendbuf, recvbuf, bufsize, ncclFloat, ncclSum, comms_[i], streams_[i]));
    }
    groupEnd();
    //std::cerr << "scatterReduce submitted" << std::endl;
    synchronizeAll();
    //std::cerr << "scatterReduce completed" << std::endl;
  }

  void allGather() override {
    size_t begin, end;
    groupStart();
    for(int i = 0; i < graphs_.size(); ++i) {
      std::tie
      (begin, end) = shardRange(i);
      //std::cerr << "[" << mpi_->to_string() << "] allGather " << begin << " " << end << std::endl;

      auto vals = graphs_[i]->params()->vals();
      const auto* sendbuf = vals->subtensor(begin, end-begin)->data();
      void*       recvbuf = vals->data();
      size_t      bufsize = shardSize();

      NCCLCHECK(ncclAllGather(sendbuf, recvbuf, bufsize, ncclFloat, comms_[i], streams_[i]));
    }
    groupEnd();
    synchronizeAll();
  }

  // swap paramShards worker[0].device[0] with a sharded set
  // This is used for the smoothed parameters, and also for persisting optimizer state.
  void swapParams(const std::vector<Tensor>& paramShards) override {
    ABORT_IF(mpi_ != nullptr, "swapParams() support for MPI is not yet implemented");
    // Update all graphs with parameter shard

    auto gather = [this, paramShards](size_t idx, size_t begin, size_t end) {
      // copy parameter shard to each graph, apart from last graph
      for(int i = 0; i < graphs_.size() - 1; ++i) {
        auto subParam
            = graphs_[i]->params()->vals()->subtensor(begin, end-begin);
        subParam->copyFrom(paramShards[idx]);
      }

      // back-up shard from last graph
      auto subParamLast
          = graphs_.back()->params()->vals()->subtensor(begin, end-begin);
      paramShards[idx]->copyFrom(subParamLast);

      auto subParamFirst
          = graphs_.front()->params()->vals()->subtensor(begin, end-begin);
      subParamLast->copyFrom(subParamFirst);
    };

    // execute for each shard
    foreach(gather);
  }

#if 0
  void pushParams(std::vector<Tensor>& paramShards) override {
    ABORT_IF(mpi_ != nullptr, "pushParams() support for MPI is not yet implemented");
    // Copy paramter shard from i-th graph to shard paramShards[i].
    // Graphs and shards with the same index live on the same device.

    auto copy = [this, paramShards](size_t idx, size_t begin, size_t end) {
      // copy parameter shard to each graph
      auto subParam
          = graphs_[idx]->params()->vals()->subtensor(begin, paramShards[idx]->size());
      paramShards[idx]->copyFrom(subParam);
    };

    foreach(copy);
  }

  void pullParams(const std::vector<Tensor>& paramShards) override {
    ABORT_IF(mpi_ != nullptr, "pullParams() support for MPI is not yet implemented");
    // Update all graphs with parameter shard

    auto gather = [this, paramShards](size_t idx, size_t begin, size_t end) {
      // copy parameter shard to each graph
      for(auto graph : graphs_) {
        auto subParam
            = graph->params()->vals()->subtensor(begin, paramShards[idx]->size());
        subParam->copyFrom(paramShards[idx]);
      }
    };
    foreach(gather);
  }
#endif

  // Doesn't work yet with NCCL
  // void pushParams(std::vector<Tensor>& params) {
  //   // Copy paramter shard from i-th graph to shard params[i].
  //   // Graphs and shards with the same index live on the same device.

  //   int pos = 0;
  //   for(int i = 0; i < graphs_.size(); ++i) {
  //     auto subParam = graphs_[i]->params()->vals()->subtensor(pos,
  //                                                             params[i]->size());
  //     groupStart();
  //     ncclBroadcast(subParam->data(),
  //                   params[i]->data(),
  //                   params[i]->size(),
  //                   ncclFloat,
  //                   0,
  //                   comms_[i],
  //                   streams_[i]);
  //     groupEnd();
  //     pos += params[i]->size();
  //   }
  //   synchronizeAll();
  // }

  // void pullParams(const std::vector<Tensor>& params) {
  //   // Update all graphs with parameter shard

  //   int totalSize = graphs_[0]->params()->vals()->size();
  //   int shardSize = ceil(totalSize / (float)graphs_.size());

  //   groupStart();
  //   for(int i = 0; i < graphs_.size(); ++i) {

  //     const void* sendbuff = (const void*)params[i]->data();
  //     void* recvbuff = (void*)graphs_[i]->params()->vals()->data();

  //     ncclAllGather(sendbuff,
  //                   recvbuff,
  //                   shardSize,
  //                   ncclFloat,
  //                   comms_[i],
  //                   streams_[i]);
  //   }
  //   groupEnd();

  //   synchronizeAll();
  // }
};

//Ptr<ICommunicator> newNCCLCommunicator(const std::vector<Ptr<ExpressionGraph>>& graphs, Ptr<IMPIWrapper> mpi) {
//  return New<NCCLCommunicator>(graphs, mpi);
//}

}  // namespace marian
