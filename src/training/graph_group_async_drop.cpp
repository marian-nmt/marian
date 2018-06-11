#include "training/graph_group_async.h"
#include "training/graph_group_async_drop.h"

#include "functional/functional.h"
#include "tensors/tensor_operators.h"
#include "training/gradient_dropping/dropper.h"
#include "training/gradient_dropping/sparse_tensor.h"

namespace marian {

void AsyncGraphGroupDrop::fetchParams(Tensor oldParams,
                                      const std::vector<Tensor>& params,
                                      int device_id) {
  // Full fetch when fetching moving average OR still in warm-up period.
  if(&params == &paramsAvg_ || fetchStep_[device_id]++ <= dropping_warmup) {
    AsyncGraphGroup::fetchParams(oldParams, params, device_id);
    return;
  }

  std::vector<std::thread> threads;
  int pos = 0;
  for(int idx = 0; idx < devices_.size(); idx++) {
    threads.emplace_back(std::thread(
        [=](int idx, int pos) {
          auto sparseGrad = sparseGrads_[device_id][idx];
          auto sparseShard = sparseShards_[device_id][idx];

          // individual mutex per-shard
          std::lock_guard<std::mutex> guard(shardSync_[idx]);

          sparseShard->gather(params[idx]);
          sparseGrad->copyFrom(sparseShard);
          sparseGrad->scatterUpdate(
              oldParams->subtensor(pos, params[idx]->size()));
        },
        idx,
        pos));

    pos += shardSize_;
  }
  for(auto&& t : threads)
    t.join();
}

void AsyncGraphGroupDrop::pushGradients(Tensor newGrads,
                                        size_t batch_words,
                                        int device_id) {
  if(pushStep_[device_id]++ < dropping_warmup) {
    AsyncGraphGroup::pushGradients(newGrads, batch_words, device_id);
    return;
  }

  // add instead of copy?
  std::vector<std::thread> threads;
  int pos = 0;
  for(int idx = 0; idx < devices_.size(); idx++) {
    threads.emplace_back(std::thread(
        [=](int idx, int pos) {
          auto dropper = droppers_[device_id][idx];
          auto sparseGrad = sparseGrads_[device_id][idx];
          auto sparseShard = sparseShards_[device_id][idx];
          auto tensor = newGrads->subtensor(pos, grads_[idx]->size());
          // individual mutex per-shard
          std::lock_guard<std::mutex> guard(shardSync_[idx]);

          // drop the gradients
          dropper->dropGraph(
              tensor, sparseGrad, droping_rate, dropping_momentum);

          // send the sharded sparse tensor
          sparseShard->copyFrom(sparseGrad);

          // convert back to dense, store it in grads_[idx]
          // sparseShard indices is equal to the indices of the sparse gradient
          // which will be used for sparse fetching
          sparseShard->toDense(grads_[idx]);

          // optimize
          if(scaleLearningRate_) {
            shardOpt_[idx]->update(
                params_[idx], grads_[idx], batch_words / avgBatchWords_);
          } else {
            shardOpt_[idx]->update(params_[idx], grads_[idx]);
          }

          if(movingAvg_)
            updateMovingAverage(
                paramsAvg_[idx], params_[idx], scheduler_->numberOfBatches());
        },
        idx,
        pos));

    pos += shardSize_;
  }
  for(auto&& t : threads)
    t.join();
}

void AsyncGraphGroupDrop::init(Ptr<data::Batch> batch) {
  AsyncGraphGroup::init(batch);
  // extra inits for gradient dropping
  if(drop_first) {
    for(int i = 0; i < devices_.size(); i++) {
      // warm-up counter
      fetchStep_.push_back(0);
      pushStep_.push_back(0);
      fetch_ready.push_back(false);

      // Size of the sparse tensor
      int totalSize = graphs_[0]->params()->vals()->size();
      int sparseCap = totalSize * 1.2 * (1.0 - droping_rate);

      // prepare droppers
      std::vector<GradientDrop> tmpDropper;
      for(auto device : devices_)
        tmpDropper.push_back(PrepareGradientDrop(graphs_[i]->getDevice()));
      droppers_.push_back(tmpDropper);

      // sparsetensor to store sparsified gradients per-device per-shard
      std::vector<SparseTensor> tmp;
      for(int j = 0; j < devices_.size(); j++)
        tmp.push_back(SparseTensor(new SparseTensorBase(
            sparseCap / devices_.size(), graphs_[i]->getBackend())));
      sparseGrads_.push_back(tmp);

      std::vector<SparseTensor> tmp2;
      for(int j = 0; j < devices_.size(); j++)
        tmp2.push_back(SparseTensor(new SparseTensorBase(
            sparseCap / devices_.size(), graphs_[j]->getBackend())));
      sparseShards_.push_back(tmp2);
    }
    drop_first = false;
  }
}
}
