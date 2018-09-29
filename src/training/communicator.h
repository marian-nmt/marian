#pragma once

// clang-format off
#include "graph/expression_graph.h"
#include "functional/functional.h"
#include "tensors/tensor_operators.h"
#include "optimizers/optimizers.h"
#if MPI_FOUND
#include "mpi.h"
#endif
// clang-format on

namespace marian {

struct/*interface*/ IMPIWrapper; // @TODO: Should we use a separate header, or move this declaration up here?

// This interface implements the cross-GPU operations for distributed training within a single box.
class ICommunicator {
protected:
  const std::vector<Ptr<ExpressionGraph>> graphs_;

public:
  ICommunicator(const std::vector<Ptr<ExpressionGraph>>& graphs)
      : graphs_(graphs) {}

  virtual ~ICommunicator() {}

  // helper to apply a function to each graph or shard, in parallel threads
  virtual void foreach(const std::function<void(size_t /*index*/, size_t /*shardBegin*/, size_t /*shardEnd*/)>& func, bool parallel = true) const = 0;

  // @TODO: all of these are const, no?
  virtual void scatterReduce() = 0; // reduce param gradients and scatter into gradient shards
  virtual void allGather() = 0;     // redistribute value shards into param values

  //virtual void pushParams(std::vector<Tensor>& paramShards) = 0;
  //virtual void pullParams(const std::vector<Tensor>& paramShards) = 0;
  virtual void swapParams(const std::vector<Tensor>& paramShards) = 0;

  virtual void scatterState(const std::vector<float>& data, const OptimizerBase::ScatterStateSetFunc& setFn) const = 0;
  virtual std::vector<float> gatherState(const OptimizerBase::GatherStateGetFunc& getFn) const = 0;
};

// @TODO: add comment to say what this is and when it is used
class DefaultCommunicator : public ICommunicator {
private:
  std::vector<Ptr<TensorAllocator>> paramsAllocs_;
  std::vector<Tensor> tmpTensors_;

  void init() {
    if(tmpTensors_.size() == 0) {
      int totalSize = (int)graphs_[0]->params()->vals()->size();
      int shardSize = (int)ceil(totalSize / (float)graphs_.size());

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

public:
  DefaultCommunicator(const std::vector<Ptr<ExpressionGraph>>& graphs, Ptr<IMPIWrapper> mpi)
      : ICommunicator(graphs) {
    ABORT_IF(mpi != nullptr, "DefaultCommunicator support for MPI is not yet implemented");
  }

  ~DefaultCommunicator() override {}

  void foreach(const std::function<void(size_t, size_t /*shardBegin*/, size_t /*shardEnd*/)>& func, bool parallel = true) const override {
    parallel &= graphs_.size() > 1;

    size_t totalSize = graphs_[0]->params()->vals()->size();
    size_t shardSize = (size_t)ceil(totalSize / (float)graphs_.size());

    size_t pos = 0;
    std::vector<std::thread> group;
    // iterate over all shards
    for(size_t idx = 0; idx < graphs_.size(); ++idx) {
      size_t size = std::min(shardSize, totalSize);

      if (parallel)
        group.emplace_back(func, idx, pos, pos+size);
      else
        func(idx, pos, pos+size);

      pos += size;
      totalSize -= size;
    }
    for(auto& t : group) // (note: group is empty is not parallel)
      t.join();
  }

  void scatterReduce() override {
    init();

    int totalSize = (int)graphs_[0]->params()->vals()->size();
    int shardSize = (int)ceil(totalSize / (float)graphs_.size());

    // Gather gradients from different devices into current gradient shards
    auto scatter = [this, shardSize](size_t idx, size_t begin, size_t end) {
      auto curGrad = graphs_[idx]->params()->grads()->subtensor(begin, end-begin);

      // collect and sum gradients
      for(auto graph : graphs_) {
        if(graph != graphs_[idx]) {
          auto subGrad = graph->params()->grads()->subtensor(begin, end - begin);
          tmpTensors_[idx]->copyFrom(subGrad);

          using namespace functional;
          Element(_1 = _1 + _2, curGrad, tmpTensors_[idx]);
        }
      }
    };

    foreach(scatter);
  }

  void allGather() override {
    int totalSize = (int)graphs_[0]->params()->vals()->size();
    int shardSize = (int)ceil(totalSize / (float)graphs_.size());

    // Update all graphs with parameter shard
    auto gather = [this, shardSize](size_t idx, size_t begin, size_t end) {
      auto getShard = [&](Ptr<ExpressionGraph> graph) {
        return graph->params()->vals()->subtensor(begin, end-begin);
      };
      auto curShard = getShard(graphs_[idx]);

      // Copy parameter shard to each graph
      for(auto graph : graphs_) {
        if(graph != graphs_[idx]) {
          auto subShard = getShard(graph);
          subShard->copyFrom(curShard);
        }
      }
    };

    foreach(gather);
  }

#if 0
  void pushParams(std::vector<Tensor>& paramShards) override {
    // Copy paramter shard from i-th graph to shard paramShards[i].
    // Graphs and shards with the same index live on the same device.

    auto copy = [this, paramShards](size_t idx, size_t begin, size_t end) {
      ABORT_IF(end-begin != paramShards[idx]->size(), "inconsistent shard size (pushParams [{}], {} vs {})??", idx, end-begin, paramShards[idx]->size());
      // Copy parameter shard to each graph
      auto subParam
          = graphs_[idx]->params()->vals()->subtensor(begin, paramShards[idx]->size());
      paramShards[idx]->copyFrom(subParam);
    };

    foreach(copy);
  }

  void pullParams(const std::vector<Tensor>& paramShards) override {
    // Update all graphs with parameter shard

    auto gather = [this, paramShards](size_t idx, size_t begin, size_t end) {
      ABORT_IF(end-begin != paramShards[idx]->size(), "inconsistent shard size (pullParams, [{}], {} vs {})??", idx, end-begin, paramShards[idx]->size());
      // Copy parameter shard to each graph
      for(auto graph : graphs_) {
        auto subParam
            = graph->params()->vals()->subtensor(begin, paramShards[idx]->size());
        subParam->copyFrom(paramShards[idx]);
      }
    };
    foreach(gather);
  }
#endif

  void swapParams(const std::vector<Tensor>& paramShards) override {
    // Update all graphs with parameter shard
    ABORT_IF(graphs_.size() < 2, "Swap requires at least two graphs");

    auto gather = [this, paramShards](size_t idx, size_t begin, size_t end) {
      ABORT_IF(end-begin != paramShards[idx]->size(), "inconsistent shard size (swapParams, [{}], {} vs {})??", idx, end-begin, paramShards[idx]->size());
      // Copy parameter shard to each graph, apart from last graph
      for(int i = 0; i < (int)graphs_.size() - 1; ++i) {
        auto subParam
            = graphs_[i]->params()->vals()->subtensor(begin, paramShards[idx]->size());
        subParam->copyFrom(paramShards[idx]);
      }

      // Back-up shard from last graph
      auto subParamLast =
          graphs_.back()->params()->vals()->subtensor(begin, paramShards[idx]->size());
      paramShards[idx]->copyFrom(subParamLast);

      auto subParamFirst
          = graphs_[0]->params()->vals()->subtensor(begin, paramShards[idx]->size());
      subParamLast->copyFrom(subParamFirst);
    };
    // Execute for each shard
    foreach(gather);
  }

  void scatterState(const std::vector<float>& data, const OptimizerBase::ScatterStateSetFunc& setFn) const override {
    size_t dataSize = data.size();
    size_t numLocalDevices = graphs_.size();
    size_t shardSize = (dataSize + numLocalDevices - 1) / numLocalDevices;// (size_t)(ceil(dataSize / (float)numLocalDevices));
    for(size_t localDeviceIndex = 0; localDeviceIndex < numLocalDevices; localDeviceIndex++) {
      size_t begin = localDeviceIndex * shardSize;
      size_t end   = std::min(begin + shardSize, dataSize);
      setFn(localDeviceIndex, data.begin() + begin, data.begin() + end);
    }
  }

  std::vector<float> gatherState(const OptimizerBase::GatherStateGetFunc& getFn) const override {
    std::vector<float> data; // we know the size here
    for (size_t localDeviceIndex = 0; localDeviceIndex < graphs_.size(); localDeviceIndex++) {
      std::vector<float> tmp = getFn(localDeviceIndex);
      data.insert(data.end(), tmp.begin(), tmp.end());
    }
    ABORT_IF(data.size() != graphs_[0]->params()->vals()->size(), "gathering wrong amount of data??");
    return data;
  }
};

Ptr<ICommunicator> createCommunicator(
    const std::vector<Ptr<ExpressionGraph>>& graphs,
    bool noNccl, Ptr<IMPIWrapper> mpi);

// Abstracts MPI operations, allowing alternative implementations (specifically fake (for debugging) and NCCL.
// This implements the MPI APIs we use here, with the following modifications:
//  * aborts with ABORT() instead of returning an error
//  * swapped out some strange MPI-specific data types to more correct C++ ones where appropriate
#if MPI_FOUND
#else
enum MPI_Comm { MPI_COMM_WORLD };
enum MPI_Datatype { MPI_FLOAT, MPI_UNSIGNED_LONG_LONG, MPI_UNSIGNED_LONG };
enum MPI_Op { MPI_SUM };
struct MPI_Status { int MPI_SOURCE; };
#define MPI_ANY_SOURCE ((size_t)-2)
#define MPI_STATUS_IGNORE ((MPI_Status*)nullptr)
#endif
struct/*interface*/ IMPIWrapper
{
  virtual size_t myRank() const = 0;
  virtual size_t commWorldSize() const = 0;
  virtual void barrier(MPI_Comm comm = MPI_COMM_WORLD) const = 0;
  virtual void bCast(void* buf, size_t count, MPI_Datatype datatype, size_t rootRank = 0, MPI_Comm comm = MPI_COMM_WORLD) const = 0;
  virtual void sSend(void* buf, size_t count, MPI_Datatype datatype, size_t destRank, int tag, MPI_Comm comm = MPI_COMM_WORLD) const = 0;
  virtual void recv(void* buf, size_t count, MPI_Datatype datatype, size_t sourceRank, int tag, MPI_Comm comm = MPI_COMM_WORLD, MPI_Status* status = MPI_STATUS_IGNORE) const = 0;
  virtual void allReduce(const void* sendbuf, void* recvbuf, size_t count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm = MPI_COMM_WORLD) const = 0;
  virtual void finalize() = 0;
  static const size_t RECV_ANY_SOURCE = (size_t)MPI_ANY_SOURCE;
  // helper templates
private:
  static MPI_Datatype getDataType(const float*) { return MPI_FLOAT; }
  static MPI_Datatype getDataType(const unsigned long*) { return MPI_UNSIGNED_LONG; }
  static MPI_Datatype getDataType(const unsigned long long*) { return MPI_UNSIGNED_LONG_LONG; }
public:
  template<typename T>
  void bCast(std::vector<T>& v, size_t rootRank = 0, MPI_Comm comm = MPI_COMM_WORLD) {
    size_t vecLen = v.size();
    bCast(&vecLen, 1, getDataType(&vecLen), rootRank, comm);
    v.resize(vecLen);
    bCast(v.data(), v.size(), getDataType(v.data()), rootRank, comm);
  }
  std::string to_string() { // helper to identify the node in logs
      return "worker " + std::to_string(myRank()) + " out of " + std::to_string(commWorldSize());
  }
};

Ptr<IMPIWrapper> initMPI(bool multiThreaded);
void finalizeMPI(Ptr<IMPIWrapper>&&);

}  // namespace marian
