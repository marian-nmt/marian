#pragma once

#include <memory>
#include <iostream>

#include "expressions.h"
#include "thrust_functions.h"
#include "tensor_operators.h"

namespace marian {

class SGD {
  public:
    SGD(Expr& cost_func, Expr& inX, Expr& inY,
        const std::vector<Expr*> params, float eta,
        std::vector<float>& xData, size_t numFeatures,
        std::vector<float>& yData, size_t numClasses,
        size_t epochs, size_t batchSize);

    void Run();

    void PrepareBatch(size_t startId, size_t endId, Tensor& xt, Tensor& yt);

    void UpdateModel();

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
