// @TODO: why is this a .cu file? It does not contain any CUDA kernels.

// clang-format off
#include "training/communicator.h"
#include "functional/functional.h"
#include "tensors/tensor_operators.h"
// clang-format on

#ifdef USE_NCCL
#include "cuda_runtime.h"
#include "nccl.h"
#endif

namespace marian {

#ifdef USE_NCCL

#define NCCLCHECK(cmd) do {                         \
    LOG(info, "[nccl] {}", #cmd); \
    ncclResult_t r = cmd;                             \
    LOG(info, "[nccl] {} -> {}", #cmd, r); \
    ABORT_IF(r != ncclSuccess, "Failed, NCCL error {} '{}'",             \
          #cmd, ncclGetErrorString(r));   \
  } while(0)

class NCCLCommunicator : public Communicator {
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

  size_t myRankWithMPI(size_t index) const { // map local device index to a global MPI rank
    return mpi_->myRank() * devices_.size() + index;
  }

  size_t numRanksWithMPI() const { // map local device index to a global MPI rank
    return mpi_->commWorldSize() * devices_.size();
  }

public:
  // a NCCLCommunicator is bound to a set of graphs, one per GPU device
  // If MPI is used, then each worker has an instance of this class for its specific
  // set of GPU devices, which are communicating with each other.
  NCCLCommunicator(const std::vector<Ptr<ExpressionGraph>>& graphs, Ptr<IMPIWrapper> mpi)
      : Communicator(graphs),
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
      ncclUniqueId uniqueId;
      if (mpi->myRank() == 0) ncclGetUniqueId(&uniqueId);
      LOG(info, "before bcast: unique id = {}", std::string(uniqueId.internal, NCCL_UNIQUE_ID_BYTES));
      mpi_->bCast((void*)&uniqueId, sizeof(uniqueId), MPI_BYTE, 0);
      LOG(info, "unique id = {}", std::string(uniqueId.internal, NCCL_UNIQUE_ID_BYTES));

      // initialize NCCL with group API
      NCCLCHECK(ncclGroupStart());
      for (int i = 0; i < devices_.size(); i++) {
        cudaSetDevice(devices_[i]);
        //LOG(info, "ncclCommInitRank {}, {}", numRanksWithMPI(), myRankWithMPI(i));
        NCCLCHECK(ncclCommInitRank(&comms_[i], numRanksWithMPI(), uniqueId, myRankWithMPI(i)));
        LOG(info, "done ncclCommInitRank {}, {}", numRanksWithMPI(), myRankWithMPI(i));
      }
      NCCLCHECK(ncclGroupEnd());
    }
    // without MPI, we have a handy convenience version to initialize
    else {
      NCCLCHECK(ncclCommInitAll(comms_.data(), devices_.size(), devices_.data()));
    }
    LOG(info, "done constructing NCCLCommunicator");
  }

  ~NCCLCommunicator() override {
    for(int i = 0; i < devices_.size(); ++i) {
      cudaSetDevice(devices_[i]);
      cudaStreamDestroy(streams_[i]);
      ncclCommDestroy(comms_[i]);
    }
  }

  void allReduceGrads() override {
    // @BUGBUG: This function is untested with MPI. If we don't need it, remove.
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

  void reduceGrads(size_t root) override {
    ncclGroupStart(); // this will aggregate across nodes and across devices inside nodes (we only loop over the local devices here)
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
    ABORT_IF(mpi_ != nullptr, "allReduceGrads() support for MPI is not yet implemented");
    int totalSize = graphs_[0]->params()->vals()->size();
    int shardSize = ceil(totalSize / (float)graphs_.size());

    int pos = 0;

    ncclGroupStart();
    for(int i = 0; i < graphs_.size(); ++i) {
      int size = std::min(shardSize, totalSize);

      const void* sendbuff = (const void*)graphs_[i]->params()->grads()->data();
      auto subgrad = graphs_[i]->params()->grads()->subtensor(pos, size);
      void* recvbuff = subgrad->data();

      ncclReduceScatter(sendbuff,
                        recvbuff,
                        shardSize,
                        ncclFloat,
                        ncclSum,
                        comms_[i],
                        streams_[i]);

      pos += size;
      totalSize -= size;
    }
    ncclGroupEnd();

    synchronizeAll();
  }

  void allGather(bool vals) override {
    ABORT_IF(mpi_ != nullptr, "allReduceGrads() support for MPI is not yet implemented");
    int totalSize = graphs_[0]->params()->vals()->size();
    int shardSize = ceil(totalSize / (float)graphs_.size());

    int pos = 0;

    ncclGroupStart();
    for(int i = 0; i < graphs_.size(); ++i) {
      int size = std::min(shardSize, totalSize);

      auto tensor = vals ? graphs_[i]->params()->vals() : graphs_[i]->params()->grads();
      auto subparam = tensor->subtensor(pos, size);
      const void* sendbuff = (const void*)subparam->data();
      void* recvbuff = (void*)tensor->data();

      ncclAllGather(
          sendbuff, recvbuff, shardSize, ncclFloat, comms_[i], streams_[i]);

      pos += size;
      totalSize -= size;
    }
    ncclGroupEnd();

    synchronizeAll();
  }

  void swapParams(const std::vector<Tensor>& params) override {
    ABORT_IF(mpi_ != nullptr, "allReduceGrads() support for MPI is not yet implemented");
    // Update all graphs with parameter shard
    ABORT_IF(graphs_.size() < 2, "Swap requires at least two graphs");

    auto gather = [this, params](size_t idx, int pos) {
      // copy parameter shard to each graph, apart from last graph
      for(int i = 0; i < graphs_.size() - 1; ++i) {
        auto subParam
            = graphs_[i]->params()->vals()->subtensor(pos, params[idx]->size());
        subParam->copyFrom(params[idx]);
      }

      // back-up shard from last graph
      auto subParamLast = graphs_.back()->params()->vals()->subtensor(
          pos, params[idx]->size());
      params[idx]->copyFrom(subParamLast);

      auto subParamFirst
          = graphs_[0]->params()->vals()->subtensor(pos, params[idx]->size());
      subParamLast->copyFrom(subParamFirst);
    };

    // execute for each shard
    this->foreach(gather);
  }

  void pushParams(std::vector<Tensor>& params) override {
    ABORT_IF(mpi_ != nullptr, "allReduceGrads() support for MPI is not yet implemented");
    // Copy paramter shard from i-th graph to shard params[i].
    // Graphs and shards with the same index live on the same device.

    auto copy = [this, params](size_t idx, int pos) {
      // copy parameter shard to each graph
      auto subParam
          = graphs_[idx]->params()->vals()->subtensor(pos, params[idx]->size());
      params[idx]->copyFrom(subParam);
    };

    this->foreach(copy);
  }

  void pullParams(const std::vector<Tensor>& params) override {
    ABORT_IF(mpi_ != nullptr, "allReduceGrads() support for MPI is not yet implemented");
    // Update all graphs with parameter shard

    auto gather = [this, params](size_t idx, int pos) {
      // copy parameter shard to each graph
      for(auto graph : graphs_) {
        auto subParam
            = graph->params()->vals()->subtensor(pos, params[idx]->size());
        subParam->copyFrom(params[idx]);
      }
    };
    this->foreach(gather);
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
#endif

Ptr<Communicator> newNCCLCommunicator(const std::vector<Ptr<ExpressionGraph>>& graphs, Ptr<IMPIWrapper> mpi) {
  return New<NCCLCommunicator>(graphs, mpi);
}

}  // namespace marian
