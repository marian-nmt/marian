#include "training/communicator.h"
#include "functional/functional.h"
#include "tensors/tensor_operators.h"

#ifdef USE_NCCL
#include "cuda_runtime.h"
#include "nccl.h"
#endif

namespace marian {

#ifdef USE_NCCL
class NCCLCommunicator : public Communicator {
private:
  std::vector<ncclComm_t> comms_;
  std::vector<cudaStream_t> streams_;
  std::vector<int> devices_;

  void synchronizeAll() {
    for(int i = 0; i < graphs_.size(); ++i) {
      cudaSetDevice(devices_[i]);
      cudaStreamSynchronize(streams_[i]);
    }
  }

public:
  NCCLCommunicator(const std::vector<Ptr<ExpressionGraph>>& graphs)
   : Communicator(graphs),
     comms_(graphs.size()),
     streams_(graphs.size()),
     devices_(graphs.size())
  {
    LOG(info, "[comm] Using NCCL library for GPU communication");

    for(int i = 0; i < graphs_.size(); ++i) {
      auto device = graphs_[i]->getBackend()->getDevice();

      ABORT_IF(device.type != DeviceType::gpu,
               "NCCL communicator can only be used with GPUs");

      devices_[i] = device.no;
      cudaSetDevice(devices_[i]);
      cudaStreamCreate(&streams_[i]);
    }

    ncclCommInitAll(comms_.data(), devices_.size(), devices_.data());
  }

  ~NCCLCommunicator() override {
    for(int i = 0; i < devices_.size(); ++i) {
      cudaSetDevice(devices_[i]);
      cudaStreamDestroy(streams_[i]);
      ncclCommDestroy(comms_[i]);
    }
  }

  void scatterReduce() override {
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

  void allGather() override {
    int totalSize = graphs_[0]->params()->vals()->size();
    int shardSize = ceil(totalSize / (float)graphs_.size());

    int pos = 0;

    ncclGroupStart();
    for(int i = 0; i < graphs_.size(); ++i) {
      int size = std::min(shardSize, totalSize);

      auto subparam = graphs_[i]->params()->vals()->subtensor(pos, size);
      const void* sendbuff = (const void*)subparam->data();
      void* recvbuff = (void*)graphs_[i]->params()->vals()->data();

      ncclAllGather(sendbuff,
                    recvbuff,
                    shardSize,
                    ncclFloat,
                    comms_[i],
                    streams_[i]);

      pos += size;
      totalSize -= size;
    }
    ncclGroupEnd();

    synchronizeAll();
  }

  void swapParams(const std::vector<Tensor>& params) override {
    // Update all graphs with parameter shard
    ABORT_IF(graphs_.size() < 2, "Swap requires at least two graphs");

    auto gather = [this, params](size_t idx, int pos) {
      // copy parameter shard to each graph, apart from last graph
      for(int i = 0; i < graphs_.size() - 1; ++i) {
        auto subParam = graphs_[i]->params()->vals()->subtensor(pos, params[idx]->size());
        subParam->copyFrom(params[idx]);
      }

      // back-up shard from last graph
      auto subParamLast = graphs_.back()->params()->vals()->subtensor(pos, params[idx]->size());
      params[idx]->copyFrom(subParamLast);

      auto subParamFirst = graphs_[0]->params()->vals()->subtensor(pos, params[idx]->size());
      subParamLast->copyFrom(subParamFirst);
    };

    // execute for each shard
    this->foreach(gather);
  }

  void pushParams(std::vector<Tensor>& params) override {
    // Copy paramter shard from i-th graph to shard params[i].
    // Graphs and shards with the same index live on the same device.

    auto copy = [this, params](size_t idx, int pos) {
      // copy parameter shard to each graph
      auto subParam = graphs_[idx]->params()->vals()->subtensor(pos, params[idx]->size());
      params[idx]->copyFrom(subParam);
    };

    this->foreach(copy);
  }

  void pullParams(const std::vector<Tensor>& params) override {
    // Update all graphs with parameter shard

    auto gather = [this, params](size_t idx, int pos) {
      // copy parameter shard to each graph
      for(auto graph : graphs_) {
        auto subParam = graph->params()->vals()->subtensor(pos, params[idx]->size());
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
  //     auto subParam = graphs_[i]->params()->vals()->subtensor(pos, params[i]->size());
  //     ncclGroupStart();
  //     ncclBroadcast((const void*)subParam->data(),
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

Ptr<Communicator> createCommunicator(const std::vector<Ptr<ExpressionGraph>>& graphs, bool noNccl) {
#ifdef USE_NCCL
  if(noNccl) {
    LOG(warn, "[comm] NCCL communicator overridden");
    return New<DefaultCommunicator>(graphs);
  }

  // if at least one of the devices is not a gpu, fall-back to default
  for(auto& graph : graphs) {
    if(graph->getBackend()->getDevice().type == DeviceType::cpu) {
      return New<DefaultCommunicator>(graphs);
    }
  }

  size_t d = graphs.size();
  if((d & (d - 1)) != 0) {
    LOG(warn, "[comm] Number of devices {} is not a power of 2 and communication might be slow with NCCL", d);
    LOG(warn, "[comm] You can switch off NCCL with --no-nccl option", d);
  }

  return New<NCCLCommunicator>(graphs);
#else
  return New<DefaultCommunicator>(graphs);
#endif
}

}
