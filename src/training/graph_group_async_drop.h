#pragma once

#include "training/graph_group_async.h"

#include "training/dropper.h"
#include "training/sparse_tensor.h"

namespace marian {

class AsyncGraphGroupDrop : public AsyncGraphGroup {
  std::vector<int> fetchStep_;
  std::vector<int> pushStep_;
  const int FETCH_WARMUP = 100;
  const int PUSH_WARMUP = 100;

  bool drop_first = 1;
  float droping_rate;
  std::vector<GradientDrop> pushDropper_;
  std::vector<std::vector<GradientDrop>> fetchDropper;

  std::vector<SparseTensor> pushSparseGradient_;
  std::vector<SparseTensor> pushShardedSparseGradient_;

  std::vector<SparseTensor> fetchSparseGradient_;
  std::vector<std::vector<SparseTensor>> fetchShardedSparseGradient_;

  std::vector<Tensor> paramsDelta_;
  std::vector<std::vector<Tensor>> paramsLocal_;

  std::vector<Ptr<TensorAllocator>> allocators;

  Tensor newTensor(int size, int device);

protected:
  void init(Ptr<data::Batch> batch);
  void pushGradients(Tensor newGrads, size_t batch_words, int device_id);
  void fetchParams(Tensor oldParams,
                   const std::vector<Tensor>& params,
                   int device_id);

public:
  AsyncGraphGroupDrop(Ptr<Config> options)
      : AsyncGraphGroup(options),
        droping_rate{options->get<float>("gradient-dropping")} {}
};
}
