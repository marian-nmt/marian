// Note: This must only be included if defined(CUDA_FOUND) && defined(USE_NCCL)
// clang-format off
#include "training/communicator.h"
//#include "functional/functional.h"
//#include "tensors/tensor_operators.h"
// clang-format on
 
#include "cuda_runtime.h"
#include "nccl.h"

namespace marian {

#define NCCLCHECK(cmd) do {                         \
    /*LOG(info, "[nccl] {}", #cmd);*/ \
    ncclResult_t r = cmd;                             \
    /*LOG(info, "[nccl] {} -> {}", #cmd, r);*/ \
    ABORT_IF(r != ncclSuccess, "Failed, NCCL error {} '{}'",             \
          #cmd, ncclGetErrorString(r));   \
  } while(0)

class NCCLCommunicator : public ICommunicator {
private:
  std::vector<ncclComm_t> comms_;
  std::vector<cudaStream_t> streams_;
  std::vector<int> devices_;
  Ptr<IMPIWrapper> mpi_; // non-null if multi-node

  void synchronizeAll() {
    for(int i = 0; i < graphs_.size(); ++i) {
      cudaSetDevice(devices_[i]);
      cudaStreamSynchronize(streams_[i]);
    }
  }

  size_t myRankWithMPI(size_t localDeviceIndex) const { // map local device index to a global MPI rank
    if (mpi_)
      return mpi_->myRank() * devices_.size() + localDeviceIndex;
    else
      return localDeviceIndex;
  }

  size_t numRanksWithMPI() const { // map local device index to a global MPI rank
    if (mpi_)
      return mpi_->commWorldSize() * devices_.size();
    else
      return devices_.size();
  }

  // determine the index range (begin, end) of a shard
  std::pair<size_t, size_t> shardRange(size_t localDeviceIndex) const {
    size_t numShards = numRanksWithMPI();
    size_t rank = myRankWithMPI(localDeviceIndex);
    size_t dataSize = graphs_[0]->params()->vals()->size();
    return{ dataSize * rank / numShards, dataSize * (rank + 1) / numShards };
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
      cudaSetDevice(devices_[i]);
      cudaStreamCreate(&streams_[i]);
    }

    // when using MPI, the setup is a laborious
    // cf. https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/index.html#multidevprothrd
    if (mpi_) {
      // generate NCCL unique ID at one process and broadcast to all
      ncclUniqueId uniqueId = { 0 };
      LOG(info, "[mpi rank {}] before ncclGetUniqueId", mpi_->myRank());
      if (mpi->myRank() == 0)
        NCCLCHECK(ncclGetUniqueId(&uniqueId));
      LOG(info, "[mpi rank {}] before bcast", mpi_->myRank());
      //LOG(info, "before bcast: unique id = {}", std::string(uniqueId.internal, NCCL_UNIQUE_ID_BYTES));
      mpi_->bCast(&uniqueId, NCCL_UNIQUE_ID_BYTES, MPI_BYTE, 0);
      LOG(info, "[mpi rank {}] after bcast", mpi_->myRank());
      //LOG(info, "unique id = {}", std::string(uniqueId.internal, NCCL_UNIQUE_ID_BYTES));

      // if more than one device then initialize NCCL with group API
      if (devices_.size() > 1) {
        NCCLCHECK(ncclGroupStart());
        for (int i = 0; i < devices_.size(); i++) {
          cudaSetDevice(devices_[i]);
          LOG(info, "ncclCommInitRank {}, {}", numRanksWithMPI(), myRankWithMPI(i));
          NCCLCHECK(ncclCommInitRank(&comms_[i], numRanksWithMPI(), uniqueId, myRankWithMPI(i)));
          LOG(info, "done ncclCommInitRank {}, {}", numRanksWithMPI(), myRankWithMPI(i));
        }
        NCCLCHECK(ncclGroupEnd());
        LOG(info, "[mpi rank {}] group done constructing NCCLCommunicator", mpi_->myRank());
      }
      // one device: no group API
      else {
        cudaSetDevice(devices_[0]);
        LOG(info, "[mpi rank {} of {}] ncclCommInitRank", mpi_->myRank(), mpi_->commWorldSize());
        NCCLCHECK(ncclCommInitRank(&comms_[0], mpi_->commWorldSize(), uniqueId, mpi_->myRank()));
        LOG(info, "[mpi rank {}] done constructing NCCLCommunicator", mpi_->myRank());
      }
    }
    // without MPI, we have a handy convenience version to initialize
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
      
    //int totalSize = (int)graphs_[0]->params()->vals()->size();
    //int shardSize = (int)ceil(totalSize / (float)graphs_.size());
    //
    //int pos = 0;

    std::vector<std::thread> group;
    // iterate over all shards on this worker
    size_t begin, end;
    for(size_t i = 0; i < graphs_.size(); ++i) {
      std::tie
      (begin, end) = shardRange(i);
      std::cerr << "foreach " << begin << " " << end << std::endl;
      size_t size = end-begin;
      //int size = std::min(shardSize, totalSize);

      if (parallel)
        group.emplace_back(func, i, begin, end);
      else
        func(i, begin, end);

      //pos += size;
      //totalSize -= size;
      // @TODO: safer variant is pos = totalSize * i / graphs_.size() and endpos = same for (id+1)
    }
    for(auto& t : group) // (note: group is empty is not parallel)
      t.join();
  }

  void allReduceGrads() override {
    ncclGroupStart();
    for(int i = 0; i < graphs_.size(); ++i) {
      NCCLCHECK(ncclAllReduce(graphs_[i]->params()->grads()->data(),
                              graphs_[i]->params()->grads()->data(),
                              graphs_[0]->params()->vals()->size(),
                              ncclFloat,
                              ncclSum,
                              comms_[i],
                              streams_[i]));
    }
    ncclGroupEnd();

    synchronizeAll();
  }

  // this will aggregate across nodes and across devices inside nodes (we only loop over the local devices here) into worker[0].device[0]
  // only used by graph_group_multinode_sync.cpp, which is unused now
  void reduceGrads(size_t root) override {
    ncclGroupStart();
    for(int i = 0; i < graphs_.size(); ++i) {
      NCCLCHECK(ncclReduce(graphs_[i]->params()->grads()->data(),
                           graphs_[i]->params()->grads()->data(),
                           graphs_[0]->params()->vals()->size(),
                           ncclFloat,
                           ncclSum,
                           root,
                           comms_[i],
                           streams_[i]));
    }
    ncclGroupEnd();

    synchronizeAll();
  }

  void scatterReduce() override {
    //ABORT_IF(mpi_ != nullptr, "allReduceGrads() support for MPI is not yet implemented");
    //int totalSize = graphs_[0]->params()->vals()->size();
    //int shardSize = ceil(totalSize / (float)graphs_.size());
    //
    //int pos = 0;

    size_t shardSize = shardRange(0).second;

    size_t begin, end;
    ncclGroupStart();
    for(int i = 0; i < graphs_.size(); ++i) {
      std::tie
      (begin, end) = shardRange(i);
      std::cerr << "scatterReduce " << begin << " " << end << std::endl;
      size_t size = end-begin;

      const auto* sendbuff = graphs_[i]->params()->grads()->data();
      auto subgrad = graphs_[i]->params()->grads()->subtensor(begin, size);
      auto* recvbuff = subgrad->data();

      ncclReduceScatter(sendbuff,
                        recvbuff,
                        shardSize,
                        ncclFloat,
                        ncclSum,
                        comms_[i],
                        streams_[i]);

      //pos += size;
      //totalSize -= size;
    }
    ncclGroupEnd();

    synchronizeAll();
  }

  void allGather(bool vals) override {
    //ABORT_IF(mpi_ != nullptr, "allReduceGrads() support for MPI is not yet implemented");
    //int totalSize = graphs_[0]->params()->vals()->size();
    //int shardSize = ceil(totalSize / (float)graphs_.size());
    //
    //int pos = 0;

    size_t shardSize = shardRange(0).second;

    size_t begin, end;
    ncclGroupStart();
    for(int i = 0; i < graphs_.size(); ++i) {
      std::tie
      (begin, end) = shardRange(i);
      std::cerr << "allGather " << begin << " " << end << std::endl;
      size_t size = end-begin;
      //int size = std::min(shardSize, totalSize);

      auto tensor = vals ? graphs_[i]->params()->vals() : graphs_[i]->params()->grads();
      auto subparam = tensor->subtensor(begin, size);
      const auto* sendbuff = subparam->data();
      void* recvbuff = tensor->data();

      ncclAllGather(sendbuff, recvbuff, shardSize, ncclFloat, comms_[i], streams_[i]);

      //pos += size;
      //totalSize -= size;
    }
    ncclGroupEnd();

    synchronizeAll();
  }

  // swap params worker[0].device[0] with a sharded set (in particular, that's the smoothed parameters)
  void swapParams(const std::vector<Tensor>& params) override {
    ABORT_IF(mpi_ != nullptr, "swapParams() support for MPI is not yet implemented");
    // Update all graphs with parameter shard

    auto gather = [this, params](size_t idx, size_t begin, size_t end) {
      // copy parameter shard to each graph, apart from last graph
      for(int i = 0; i < graphs_.size() - 1; ++i) {
        auto subParam
            = graphs_[i]->params()->vals()->subtensor(begin, params[idx]->size());
        subParam->copyFrom(params[idx]);
      }

      // back-up shard from last graph
      auto subParamLast
          = graphs_.back()->params()->vals()->subtensor(begin, params[idx]->size());
      params[idx]->copyFrom(subParamLast);

      auto subParamFirst
          = graphs_[0]->params()->vals()->subtensor(begin, params[idx]->size());
      subParamLast->copyFrom(subParamFirst);
    };

    // execute for each shard
    foreach(gather);
  }

  void pushParams(std::vector<Tensor>& params) override {
    ABORT_IF(mpi_ != nullptr, "pushParams() support for MPI is not yet implemented");
    // Copy paramter shard from i-th graph to shard params[i].
    // Graphs and shards with the same index live on the same device.

    auto copy = [this, params](size_t idx, size_t begin, size_t end) {
      // copy parameter shard to each graph
      auto subParam
          = graphs_[idx]->params()->vals()->subtensor(begin, params[idx]->size());
      params[idx]->copyFrom(subParam);
    };

    foreach(copy);
  }

  void pullParams(const std::vector<Tensor>& params) override {
    ABORT_IF(mpi_ != nullptr, "pullParams() support for MPI is not yet implemented");
    // Update all graphs with parameter shard

    auto gather = [this, params](size_t idx, size_t begin, size_t end) {
      // copy parameter shard to each graph
      for(auto graph : graphs_) {
        auto subParam
            = graph->params()->vals()->subtensor(begin, params[idx]->size());
        subParam->copyFrom(params[idx]);
      }
    };
    foreach(gather);
  }

  // Doesn't work yet with NCCL
  // void pushParams(std::vector<Tensor>& params) {
  //   // Copy paramter shard from i-th graph to shard params[i].
  //   // Graphs and shards with the same index live on the same device.

  //   int pos = 0;
  //   for(int i = 0; i < graphs_.size(); ++i) {
  //     auto subParam = graphs_[i]->params()->vals()->subtensor(pos,
  //     params[i]->size()); ncclGroupStart(); ncclBroadcast((const
  //     void*)subParam->data(),
  //                   (void*)params[i]->data(),
  //                   params[i]->size(),
  //                   ncclFloat,
  //                   0,
  //                   comms_[i],
  //                   streams_[i]);
  //     ncclGroupEnd();
  //     pos += params[i]->size();
  //   }
  //   synchronizeAll();
  // }

  // void pullParams(const std::vector<Tensor>& params) {
  //   // Update all graphs with parameter shard

  //   int totalSize = graphs_[0]->params()->vals()->size();
  //   int shardSize = ceil(totalSize / (float)graphs_.size());

  //   ncclGroupStart();
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
  //   ncclGroupEnd();

  //   synchronizeAll();
  // }
};

//Ptr<ICommunicator> newNCCLCommunicator(const std::vector<Ptr<ExpressionGraph>>& graphs, Ptr<IMPIWrapper> mpi) {
//  return New<NCCLCommunicator>(graphs, mpi);
//}

}  // namespace marian
