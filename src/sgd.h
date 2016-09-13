#pragma once

#include <memory>
#include <iostream>

#include "expressions.h"

namespace marian {

class SGD {
  public:
    SGD(Expr& cost_func, Expr& inX, Expr& inY, float eta, std::vector<std::vector<float>> &xData,
        std::vector<float> &yData, size_t numClasses, size_t epochs, size_t batchSize)
      : cost_function_(&cost_func),
        inX_(&inX),
        inY_(&inY),
        eta_(eta),
        xData_(xData),
        yData_(yData),
        epochs_(epochs),
        batchSize_(batchSize),
        numClasses_(numClasses) {}

    void run() {
      auto numExamples = xData_[0].size();
      Tensor xt({(int)batchSize_, (int)numExamples}, 0.0f);
      Tensor yt({(int)batchSize_, (int)numClasses_}, 0.0f);
      for (size_t numEpoch = 0; numEpoch < epochs_; ++numEpoch) {
        std::cerr << "Starting epoch #" << numEpoch << std::endl;
        size_t startId = 0;
        size_t endId = startId + batchSize_;

        while (endId < numExamples) {
          prepareBatch(startId, xt, yt);
          *inX_ = xt;
          *inY_ = yt;

          cost_function_->forward(batchSize_);
          cost_function_->backward();

          updateModel();

          startId += batchSize_;
          endId += batchSize_;
        }
      }
    }

    void prepareBatch(const size_t index, Tensor& xt, Tensor& yt) {
    }

    void updateModel() {
    }

  private:
    std::shared_ptr<Expr> cost_function_;
    std::shared_ptr<Expr> inX_;
    std::shared_ptr<Expr> inY_;
    const float eta_;
    std::vector<std::vector<float>> &xData_;
    std::vector<float> &yData_;
    const size_t epochs_;
    const size_t batchSize_;
    const size_t numClasses_;
};

} // namespace marian
