// clang-format off
#include "graph/expression_graph.h"
#include "functional/functional.h"
#include "tensors/tensor_operators.h"
// clang-format on

namespace marian {

class Communicator {
protected:
  const std::vector<Ptr<ExpressionGraph>> graphs_;

public:
  Communicator(const std::vector<Ptr<ExpressionGraph>>& graphs)
      : graphs_(graphs) {}

  virtual ~Communicator() {}

  virtual void foreach(const std::function<void(size_t, int)>& func) {
    int totalSize = (int)graphs_[0]->params()->vals()->size();
    int shardSize = (int)ceil(totalSize / (float)graphs_.size());

    int pos = 0;
    std::vector<std::thread> group;
    // iterate over all shards
    for(size_t idx = 0; idx < graphs_.size(); ++idx) {
      int size = std::min(shardSize, totalSize);

      group.emplace_back(func, idx, pos);

      pos += size;
      totalSize -= size;
    }
    for(auto& t : group)
      t.join();
  }

  virtual void scatterReduce() = 0;
  virtual void allGather() = 0;

  virtual void pushParams(std::vector<Tensor>& params) = 0;
  virtual void pullParams(const std::vector<Tensor>& params) = 0;
  virtual void swapParams(const std::vector<Tensor>& params) = 0;
};

class DefaultCommunicator : public Communicator {
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
  DefaultCommunicator(const std::vector<Ptr<ExpressionGraph>>& graphs)
      : Communicator(graphs) {}

  ~DefaultCommunicator() override {}

  void scatterReduce() override {
    init();

    int totalSize = (int)graphs_[0]->params()->vals()->size();
    int shardSize = (int)ceil(totalSize / (float)graphs_.size());

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

  void allGather() override {
    int totalSize = (int)graphs_[0]->params()->vals()->size();
    int shardSize = (int)ceil(totalSize / (float)graphs_.size());

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

  void pushParams(std::vector<Tensor>& params) override {
    // Copy paramter shard from i-th graph to shard params[i].
    // Graphs and shards with the same index live on the same device.

    auto copy = [this, params](size_t idx, int pos) {
      // copy parameter shard to each graph
      auto subParam
          = graphs_[idx]->params()->vals()->subtensor(pos, (int)params[idx]->size());
      params[idx]->copyFrom(subParam);
    };

    this->foreach(copy);
  }

  void pullParams(const std::vector<Tensor>& params) override {
    // Update all graphs with parameter shard

    auto gather = [this, params](size_t idx, int pos) {
      // copy parameter shard to each graph
      for(auto graph : graphs_) {
        auto subParam
            = graph->params()->vals()->subtensor(pos, (int)params[idx]->size());
        subParam->copyFrom(params[idx]);
      }
    };
    this->foreach(gather);
  }

  void swapParams(const std::vector<Tensor>& params) override {
    // Update all graphs with parameter shard
    ABORT_IF(graphs_.size() < 2, "Swap requires at least two graphs");

    auto gather = [this, params](size_t idx, int pos) {
      // copy parameter shard to each graph, apart from last graph
      for(int i = 0; i < (int)graphs_.size() - 1; ++i) {
        auto subParam
            = graphs_[i]->params()->vals()->subtensor(pos, (int)params[idx]->size());
        subParam->copyFrom(params[idx]);
      }

      // back-up shard from last graph
      auto subParamLast =
          graphs_.back()->params()->vals()->subtensor(pos, (int)params[idx]->size());
      params[idx]->copyFrom(subParamLast);

      auto subParamFirst
          = graphs_[0]->params()->vals()->subtensor(pos, (int)params[idx]->size());
      subParamLast->copyFrom(subParamFirst);
    };
    // execute for each shard
    this->foreach(gather);
  }
};

Ptr<Communicator> createCommunicator(
    const std::vector<Ptr<ExpressionGraph>>& graphs,
    bool noNccl = false);

}  // namespace marian
