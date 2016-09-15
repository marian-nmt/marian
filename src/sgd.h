#pragma once

#include <memory>
#include <iostream>

#include "expressions.h"
#include "thrust_functions.h"

namespace marian {

class SGD {
  public:
    SGD(Expr& cost_func, Expr& inX, Expr& inY,
        const std::vector<Expr*> params, float eta,
        std::vector<float>& xData, size_t numFeatures,
        std::vector<float>& yData, size_t numClasses,
        size_t epochs, size_t batchSize)
    : cost_function_(&cost_func),
      inX_(&inX),
      inY_(&inY),
      params_(params),
      eta_(eta),
      xData_(xData),
      numFeatures_(numFeatures),
      yData_(yData),
      numClasses_(numClasses),
      epochs_(epochs),
      batchSize_(batchSize)
  {}

    void Run() {
      size_t numExamples = xData_.size()/ numFeatures_;
      Tensor xt({(int)batchSize_, (int)numExamples}, 0.0f);
      Tensor yt({(int)batchSize_, (int)numClasses_}, 0.0f);

      for (size_t numEpoch = 0; numEpoch < epochs_; ++numEpoch) {
        std::cerr << "Starting epoch #" << numEpoch << std::endl;
        size_t startId = 0;
        size_t endId = startId + batchSize_;

        while (endId < numExamples) {
          PrepareBatch(startId, endId, xt, yt);
          *inX_ = xt;
          *inY_ = yt;

          cost_function_->forward(batchSize_);
          cost_function_->backward();

          UpdateModel();

          startId += batchSize_;
          endId += batchSize_;
        }
      }
    }

    void PrepareBatch(size_t startId, size_t endId, Tensor& xt, Tensor& yt) {
      std::vector<float> x(xData_.begin() + startId * numFeatures_,
                           xData_.begin() + endId * numFeatures_);
      std::vector<float> y(yData_.begin() + startId * numClasses_,
                           yData_.begin() + endId * numClasses_);

      xt.set(x);
      yt.set(y);
    }

    void UpdateModel() {
      for (auto& param : params_) {
        using namespace thrust::placeholders;
        Element(_1 = _1 - eta_ * _2, param->val(), param->grad());
      }
    }

  private:
    std::shared_ptr<Expr> cost_function_;
    std::shared_ptr<Expr> inX_;
    std::shared_ptr<Expr> inY_;
    std::vector<Expr*> params_;
    const float eta_;
    std::vector<float>& xData_;
    const size_t numFeatures_;
    std::vector<float>& yData_;
    const size_t numClasses_;
    const size_t epochs_;
    const size_t batchSize_;
};

} // namespace marian
