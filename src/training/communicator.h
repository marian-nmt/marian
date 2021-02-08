#pragma once

// clang-format off
#include "graph/expression_graph.h"
#include "functional/functional.h"
#include "tensors/tensor_operators.h"
#include "optimizers/optimizers.h"
#include "3rd_party/threadpool.h"
#if MPI_FOUND
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-override"
#endif
#undef HOST
#define OMPI_SKIP_MPICXX 1 // Fixes compilation with GCC8+ https://github.com/open-mpi/ompi/issues/5157
#include "mpi.h"
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
#endif
// clang-format on

#include <future>

namespace marian {

enum struct ShardingMode : size_t { global, local };

struct/*interface*/ IMPIWrapper; // @TODO: Should we use a separate header, or move this declaration up here?

ShardingMode getShardingMode(Ptr<Options> options, Ptr<IMPIWrapper> mpi);

// This interface implements the cross-GPU operations for distributed training within a single box.
class ICommunicator {
protected:
  const std::vector<Ptr<ExpressionGraph>> graphs_;

public:
  ICommunicator(const std::vector<Ptr<ExpressionGraph>>& graphs)
      : graphs_(graphs) {}

  virtual ~ICommunicator() {}

  // helper to apply a function to each local graph, in parallel threads
  template <typename ReturnType>
  using ForeachFunc = std::function<ReturnType(size_t, size_t /*shardBegin*/, size_t /*shardEnd*/)>;

  template <typename ReturnType>
  using AccFunc = std::function<void(ReturnType&, ReturnType)>;

  virtual bool foreach(const ForeachFunc<bool>& func, bool parallel = true) const = 0;
  virtual float foreach(const ForeachFunc<float>& func, AccFunc<float> acc, float init, bool parallel = true) const = 0;
  // @TODO: We probably can still share foreach() between the two implementations. Just need to move some helper functions from the .cu file.

  virtual void scatterReduceAndResetGrads() const = 0; // reduce param gradients and scatter into gradient shards
  virtual void allGatherParams() const = 0;     // redistribute value shards into param values
  virtual void broadcastParams(bool average = false) const = 0;  // average corresponding parameters across all workers
  virtual void broadcastShards(const std::vector<Ptr<OptimizerBase>>& opts, bool average = false) const = 0;

  virtual void scatterState(const io::Item& data, const OptimizerBase::ScatterStateSetFunc& setFn) const = 0;
  virtual io::Item gatherState(const OptimizerBase::GatherStateGetFunc& getFn) const = 0;
};

// Abstracts MPI operations, allowing alternative implementations (specifically fake (for debugging) and NCCL.
// This implements the MPI APIs we use here, with the following modifications:
//  * aborts with ABORT() instead of returning an error
//  * swapped out some strange MPI-specific data types to more correct C++ ones where appropriate
#if MPI_FOUND
#else
enum MPI_Comm { MPI_COMM_WORLD };
enum MPI_Datatype { MPI_FLOAT, MPI_UNSIGNED_LONG_LONG, MPI_UNSIGNED_LONG, MPI_BYTE, MPI_INT };
enum MPI_Op { MPI_SUM };
struct MPI_Status { int MPI_SOURCE; };
#define MPI_ANY_SOURCE ((size_t)-2)
#define MPI_STATUS_IGNORE ((MPI_Status*)nullptr)
#endif

struct/*interface*/ IMPIWrapper {
  virtual size_t myMPIRank() const = 0;
  virtual size_t numMPIProcesses() const = 0;
  virtual bool isMainProcess() const { return myMPIRank() == 0; }
  virtual void barrier(MPI_Comm comm = MPI_COMM_WORLD) const = 0;
  virtual void bCast(void* buf, size_t count, MPI_Datatype datatype, size_t rootRank = 0, MPI_Comm comm = MPI_COMM_WORLD) const = 0;
  virtual void sSend(void* buf, size_t count, MPI_Datatype datatype, size_t destRank, int tag, MPI_Comm comm = MPI_COMM_WORLD) const = 0;
  virtual void recv(void* buf, size_t count, MPI_Datatype datatype, size_t sourceRank, int tag, MPI_Comm comm = MPI_COMM_WORLD, MPI_Status* status = MPI_STATUS_IGNORE) const = 0;
  virtual void allReduce(const void* sendbuf, void* recvbuf, size_t count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm = MPI_COMM_WORLD) const = 0;
  virtual void finalize() = 0;
  static const size_t RECV_ANY_SOURCE = (size_t)MPI_ANY_SOURCE;

  static MPI_Datatype getDataType(const char*)               { return MPI_BYTE; }
  static MPI_Datatype getDataType(const int*)                { return MPI_INT; }
  static MPI_Datatype getDataType(const float*)              { return MPI_FLOAT; }
  static MPI_Datatype getDataType(const unsigned long*)      { return MPI_UNSIGNED_LONG; }
  static MPI_Datatype getDataType(const unsigned long long*) { return MPI_UNSIGNED_LONG_LONG; }

  void bCast(io::Item& item, size_t rootRank = 0, MPI_Comm comm = MPI_COMM_WORLD) {
    ABORT_IF(item.bytes.empty(), "Broadcasting empty item via MPI??");

    unsigned long long bytesLen = item.bytes.size();
    bCast(&bytesLen, 1, getDataType(&bytesLen), rootRank, comm);

    item.bytes.resize(bytesLen);
    bCast(item.bytes.data(), item.bytes.size(), getDataType(item.bytes.data()), rootRank, comm);

    unsigned long long shapeLen = item.shape.size();
    bCast(&shapeLen, 1, getDataType(&shapeLen), rootRank, comm);

    bCast(item.shape.data(), item.shape.size(), getDataType(item.shape.data()), rootRank, comm);

    size_t type = (size_t)item.type;
    bCast(&type, 1, getDataType(&type), rootRank, comm);
    item.type = (Type)type;
  }

  std::string idStr() const;
};

Ptr<IMPIWrapper> initMPI(bool multiThreaded);
void finalizeMPI(Ptr<IMPIWrapper>&&);

// DefaultCommunicator is used when we cannot use NCCLCommunicator, e.g. if it is not compiled in
class DefaultCommunicator : public ICommunicator {
private:
  std::vector<Ptr<TensorAllocator>> paramsAllocs_;
  std::vector<Tensor> tmpTensors_;
  mutable ThreadPool threadPool_;

  void lazyInit() {
    if(tmpTensors_.size() == 0) {
      int totalSize = (int)graphs_[0]->params()->vals()->size();
      int shardSize = (int)ceil(totalSize / (float)graphs_.size());

      int pos = 0;
      for(auto graph : graphs_) {
        int __size__ = std::min(shardSize, totalSize);

        auto paramsAlloc = New<TensorAllocator>(graph->getBackend());
        paramsAllocs_.push_back(paramsAlloc);

        paramsAlloc->reserveExact(__size__ * sizeOf(graph->getDefaultElementType()));

        Tensor tmp;

        paramsAlloc->allocate(tmp, {1, __size__}, graph->getDefaultElementType());
        tmpTensors_.push_back(tmp);

        // move to next shard
        pos += __size__;
        totalSize -= __size__;
      }
    }
  }

public:
  DefaultCommunicator(const std::vector<Ptr<ExpressionGraph>>& graphs, Ptr<IMPIWrapper> mpi)
      : ICommunicator(graphs),
        threadPool_(graphs.size(), graphs.size()) {
    ABORT_IF(mpi && mpi->numMPIProcesses() != 1, "DefaultCommunicator does not support multi-process MPI");
  }

  ~DefaultCommunicator() override {}

  size_t dataSize() const { // total number of floats that comprise the concatenated parameter and gradient vector
    return graphs_[0]->params()->vals()->size();
  }

  // determine the (max) shard size
  // All shards except the last one have this size.
  // Presently, all shards must have identical size, due to a limitation in NCCL we have not yet worked around.
  size_t shardSize() const {
    size_t numShards = graphs_.size();
    size_t size = (dataSize() + numShards - 1) / numShards;
#if 1 // for now, all shards must have the same size, since NCCL does not allow a sub-slice for the last shard
    ABORT_IF(size * numShards != dataSize(), "presently, all shards must have the same size");
#endif
    return size;
  }

  // determine the index range (begin, end) of a shard
  std::pair<size_t, size_t> localShardRange(size_t localDeviceIndex) const {
    size_t size = shardSize();
    size_t begin = localDeviceIndex * size;
    size_t end = begin + size;
    end = std::min(end, dataSize()); // clip last shard. Note: Presently this never happens, since shardSize() enforces uniform shard size.
    return { begin, end };
  }

  // @TODO: function is now the same as in NCCLCommunicator, move up to base class if possible
  template <typename Ret>
  Ret foreachAcc(const ForeachFunc<Ret>& func, const AccFunc<Ret>& acc, Ret init, bool parallel = true) const {
    parallel &= graphs_.size() > 1;

    Ret retValue = init;
    std::vector<std::future<Ret>> threadResults(graphs_.size()); // [device index]
    for(size_t i = 0; i < graphs_.size(); ++i) {
      size_t begin, end; std::tie
      (begin, end) = localShardRange(i);
      if(parallel)
        threadResults[i] = threadPool_.enqueue(func, i, begin, end);
      else
        acc(retValue, func(i, begin, end));
    }
    if(parallel)
      for(auto& task : threadResults)
        acc(retValue, task.get());

    return retValue;
  }

  float foreach(const ForeachFunc<float>& func, AccFunc<float> acc, float init, bool parallel = true) const override {
    return foreachAcc(func, acc, init, parallel);
  }

  bool foreach(const ForeachFunc<bool>& func, bool parallel = true) const override {
    AccFunc<bool> allTrue = [](bool& x, bool y) { x = x && y; };
    return foreachAcc(func, allTrue, true, parallel);
  }

  void scatterReduceAndResetGrads() const override {
    const_cast<DefaultCommunicator*>(this)->lazyInit();

    // Gather gradients from different devices into current gradient shards
    auto scatter = [this](size_t idx, size_t begin, size_t end) {
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
      return true; // dummy success
    };

    // reset gradients
    // @TODO: all the different places where gradients get reset are confusing
    auto reset = [this](size_t idx, size_t begin, size_t end) {
      auto grads = graphs_[idx]->params()->grads();
      // reset everything outside the shard that we reduce in
      if (begin > 0)
        grads->subtensor(0, begin)->set(0.f);
      if (end < grads->size())
        grads->subtensor(end, grads->size() - end)->set(0.f);

      return true; // dummy success
    };
    
    foreach(scatter);
    foreach(reset);
  }

  void allGatherParams() const override {
    // Update all graphs with parameter shard
    auto gather = [this](size_t idx, size_t begin, size_t end) {
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

      return true; // dummy success
    };

    foreach(gather);
  }

  void broadcastParams(bool average = false) const override {
    ABORT_IF(average, "Parameter averaging not implemented in DefaultCommunicator::broadcastParams");

    // Copy parameters from first graph
    auto copyFromFirst = [this](size_t idx, size_t /*begin*/, size_t /*end*/) {
      if(idx != 0)
        graphs_[idx]->params()->vals()->copyFrom(graphs_[0]->params()->vals());
      return true; // dummy success
    };

    foreach(copyFromFirst);
  }

  virtual void broadcastShards(const std::vector<Ptr<OptimizerBase>>& opts, bool average = false) const override {
    opts; average;
    ABORT("DefaultCommunicator::broadcastShards not implemented");
  }

  void scatterState(const io::Item& data, const OptimizerBase::ScatterStateSetFunc& setFn) const override {
    size_t dataSize = data.size();
    size_t numLocalDevices = graphs_.size();
    size_t shardSize = (dataSize + numLocalDevices - 1) / numLocalDevices;// (size_t)(ceil(dataSize / (float)numLocalDevices));
    for(size_t localDeviceIndex = 0; localDeviceIndex < numLocalDevices; localDeviceIndex++) {
      size_t begin = localDeviceIndex * shardSize;
      size_t end   = std::min(begin + shardSize, dataSize);
      setFn(localDeviceIndex, data.bytes.data() + begin, data.bytes.data() + end);
    }
  }

  io::Item gatherState(const OptimizerBase::GatherStateGetFunc& getFn) const override {
    io::Item data = getFn(0);
    for (size_t localDeviceIndex = 1; localDeviceIndex < graphs_.size(); localDeviceIndex++)
      data.append(getFn(localDeviceIndex));

    size_t elements = data.bytes.size() / sizeOf(data.type);
    ABORT_IF(elements != graphs_[0]->params()->vals()->size(), "gathering wrong amount of data??");
    return data;
  }
};

Ptr<ICommunicator> createCommunicator(
    const std::vector<Ptr<ExpressionGraph>>& graphs,
    bool noNccl, ShardingMode shardingMode, Ptr<IMPIWrapper> mpi);

}  // namespace marian
