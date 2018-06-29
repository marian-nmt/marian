#include "training/communicator.h"
#include "functional/functional.h"
#include "tensors/tensor_operators.h"

#ifdef USE_NCCL
#include "cuda_runtime.h"
#include "nccl.h"
#endif

namespace marian {

class DefaultCommunicator : public Communicator {
private:
  std::vector<Ptr<TensorAllocator>> paramsAllocs_;
  std::vector<Tensor> tmpTensors_;

public:
  DefaultCommunicator(const std::vector<Ptr<ExpressionGraph>>& graphs)
   : Communicator(graphs) {}

  void init() {
    if(tmpTensors_.size() == 0) {
      int totalSize = graphs_[0]->params()->vals()->size();
      int shardSize = ceil(totalSize / (float)graphs_.size());

      int pos = 0;
      for(auto graph : graphs_) {
        int __size__ = std::min(shardSize, totalSize);

        auto paramsAlloc = New<TensorAllocator>(graph->getBackend());
        paramsAllocs_.push_back(paramsAlloc);

        paramsAlloc->reserveExact(__size__ * sizeof(float));

        Tensor tmp;

        paramsAlloc->allocate(tmp, {1, __size__});
        tmpTensors_.push_back(tmp);

        // move to next shard
        pos += __size__;
        totalSize -= __size__;
      }
    }
  }

  void scatterReduce() {
    init();

    int totalSize = graphs_[0]->params()->vals()->size();
    int shardSize = ceil(totalSize / (float)graphs_.size());

    // Gather gradients from different devices into current gradient shards
    auto scatter = [this, shardSize](size_t idx, int pos) {
      auto curGrad = graphs_[idx]->params()->grads()->subtensor(pos, shardSize);

      // collect and sum gradients
      // to be replaced with ncclScatterReduce
      for(auto graph : graphs_) {
        if(graph != graphs_[idx]) {
          auto subGrad = graph->params()->grads()->subtensor(pos, shardSize);
          tmpTensors_[idx]->copyFrom(subGrad);

          using namespace functional;
          Element(_1 = _1 + _2, curGrad, tmpTensors_[idx]);
        }
      }
    };

    this->foreach(scatter);
  }

  void allGather() {
    int totalSize = graphs_[0]->params()->vals()->size();
    int shardSize = ceil(totalSize / (float)graphs_.size());

    // Update all graphs with parameter shard
    auto gather = [this, shardSize](size_t idx, int pos) {
      auto curParam = graphs_[idx]->params()->vals()->subtensor(pos, shardSize);

      // copy parameter shard to each graph
      for(auto graph : graphs_) {
        if(graph != graphs_[idx]) {
          auto subParam = graph->params()->vals()->subtensor(pos, shardSize);
          subParam->copyFrom(curParam);
        }
      }
    };

    this->foreach(gather);
  }
};

#ifdef USE_NCCL
class NCCLCommunicator : public Communicator {
private:
  std::vector<ncclComm_t> comms_;
  std::vector<cudaStream_t> streams_;
  std::vector<int> devices_;

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

  ~NCCLCommunicator() {
    for(int i = 0; i < devices_.size(); ++i) {
      cudaSetDevice(devices_[i]);
      cudaStreamDestroy(streams_[i]);
      ncclCommDestroy(comms_[i]);
    }
  }

  void scatterReduce() {
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

    for(int i = 0; i < graphs_.size(); ++i) {
      cudaSetDevice(devices_[i]);
      cudaStreamSynchronize(streams_[i]);
    }
  }

  void allGather() {
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

    for(int i = 0; i < graphs_.size(); ++i) {
      cudaSetDevice(devices_[i]);
      cudaStreamSynchronize(streams_[i]);
    }
  }
};
#endif

Ptr<Communicator> createCommunicator(const std::vector<Ptr<ExpressionGraph>>& graphs) {
#ifdef USE_NCCL
  for(auto& graph : graphs) {
    if(graph->getBackend()->getDevice().type == DeviceType::cpu) {
      return New<DefaultCommunicator>(graphs);
    }
  }
  return New<NCCLCommunicator>(graphs);
#else
  return New<DefaultCommunicator>(graphs);
#endif
}


}
