#include "graph/expression_graph.h"

namespace marian {

class Communicator {
protected:
  const std::vector<Ptr<ExpressionGraph>> graphs_;

public:
  Communicator(const std::vector<Ptr<ExpressionGraph>>& graphs)
  : graphs_(graphs) {}

  virtual ~Communicator() {}

  virtual void foreach(const std::function<void(size_t, int)>& func) {
    int totalSize = graphs_[0]->params()->vals()->size();
    int shardSize = ceil(totalSize / (float)graphs_.size());

    int pos = 0;
    std::vector<std::thread> group;
    // iterate over all shards
    for(int idx = 0; idx < graphs_.size(); ++idx) {
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

  void init();

public:
  DefaultCommunicator(const std::vector<Ptr<ExpressionGraph>>& graphs);

  void scatterReduce();
  void allGather();
  void pushParams(std::vector<Tensor>& params);
  void pullParams(const std::vector<Tensor>& params);
  void swapParams(const std::vector<Tensor>& params);
};

Ptr<Communicator> createCommunicator(const std::vector<Ptr<ExpressionGraph>>& graphs);

}
