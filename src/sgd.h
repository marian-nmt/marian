#pragma once

#include <memory>
#include <iostream>

#include "expression_graph.h"
#include "thrust_functions.h"
#include "tensor_operators.h"

namespace marian {

class SGD {
  public:
    SGD(ExpressionGraph& g, float eta,
        std::vector<float>& xData, size_t numFeatures,
        std::vector<float>& yData, size_t numClasses,
        size_t epochs, size_t batchSize);

    void Run();

  private:
    ExpressionGraph& graph_;
    const float eta_;
    std::vector<float>& xData_;
    const size_t numFeatures_;
    std::vector<float>& yData_;
    const size_t numClasses_;
    const size_t epochs_;
    const size_t maxBatchSize_;

    std::vector<size_t> CreateShuffle(size_t numExamples) const;
    void PrepareBatch(
    		size_t startId,
    		size_t endId,
    		size_t batchSize,
    		const std::vector<size_t> &shuffle,
    		Tensor& xt,
    		Tensor& yt);

    void UpdateModel();
};

} // namespace marian
