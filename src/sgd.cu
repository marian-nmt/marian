#include <algorithm>
#include <vector>
#include "sgd.h"
#include "thrust_functions.h"

using namespace std;

namespace marian {
SGD::SGD(Expr& cost_func, Expr& inX, Expr& inY,
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
  maxBatchSize_(batchSize)
{}

void SGD::Run()
{
  std::srand ( unsigned ( std::time(0) ) );

  size_t numExamples = xData_.size()/ numFeatures_;
  Tensor xt({(int)maxBatchSize_, (int)numExamples}, 0.0f);
  Tensor yt({(int)maxBatchSize_, (int)numClasses_}, 0.0f);

  vector<size_t> shuffle = CreateShuffle(numExamples);

  for (size_t numEpoch = 0; numEpoch < epochs_; ++numEpoch) {
    std::cerr << "Starting epoch #" << numEpoch << std::endl;
    size_t startId = 0;
    size_t endId = startId + maxBatchSize_;

    while (endId < numExamples) {
      PrepareBatch(startId, endId, maxBatchSize_, shuffle, xt, yt);
      *inX_ = xt;
      *inY_ = yt;

      cost_function_->forward(maxBatchSize_);
      cost_function_->backward();

      UpdateModel();

      startId += maxBatchSize_;
      endId += maxBatchSize_;
    }
  }
}

std::vector<size_t> SGD::CreateShuffle(size_t numExamples) const {
  vector<size_t> ret(numExamples);
  std::iota(ret.begin(), ret.end(), 1);
  std::random_shuffle ( ret.begin(), ret.end() );

  for (size_t i = 0; i < ret.size(); ++i) {
	  cerr << ret[i] << " ";
  }

  return ret;
}

void SGD::PrepareBatch(
		size_t startId,
		size_t endId,
		size_t batchSize,
		const std::vector<size_t> &shuffle,
		Tensor& xt,
		Tensor& yt) {

  std::vector<float> x(xData_.begin() + startId * numFeatures_,
                       xData_.begin() + endId * numFeatures_);
  std::vector<float> y(yData_.begin() + startId * numClasses_,
                       yData_.begin() + endId * numClasses_);
  /*
  std::vector<float> x(batchSize * numFeatures_);
  std::vector<float> y(batchSize * numClasses_);

  std::vector<float>::iterator startXIter = x.begin();
  std::vector<float>::iterator startYIter = y.begin();

  for (size_t i = startId; i < endId; ++i) {
    size_t startXDataId = i * numFeatures_;
    size_t startYDataId = i * numClasses_;

    size_t endXDataId = startXDataId + batchSize * numFeatures_;
    size_t endYDataId = startYDataId + batchSize * numClasses_;

    std::copy(xData_.begin() + startXDataId,
        xData_.begin() + endXDataId,
        startXIter);

    std::copy(yData_.begin() + startYDataId,
        yData_.begin() + endYDataId,
        startYIter);

    startXIter += batchSize * numFeatures_;
    startYIter += batchSize * numClasses_;
  }
  */
  xt.set(x);
  yt.set(y);
}

void SGD::UpdateModel() {
  for (auto& param : params_) {
    using namespace thrust::placeholders;
    Element(_1 = _1 - eta_ * _2, param->val(), param->grad());
  }
}

} // namespace

