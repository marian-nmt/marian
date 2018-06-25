#include "functional/functional.h"
#include "tensors/tensor_operators.h"
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
};

Ptr<Communicator> createCommunicator(const std::vector<Ptr<ExpressionGraph>>& graphs);


}
