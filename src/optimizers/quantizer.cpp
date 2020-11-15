#include <cmath>

#include "optimizers/quantizer.h"
#include "tensors/tensor_allocator.h"
#include "tensors/tensor_operators.h"

#include "functional/functional.h"

namespace marian {

/* simulate a fixed quantization for values in data.
 * For example:
 * data  = [0.96, 0.73, 0.82, 0.84, 0.42, 0.29, 0.65]
 * res   = [1   , 0.6,  0.8 , 0.8 , 0.4,  0.2 , 0.6 ]
 *
 * @param data contains the original data
 * @param res will contain the resulting quantized data. set data = res for in-place operation
 * @param numCenters the number of quantized centers in absolute. It should be 2^(bit-1)
 * @param S stores the scaling factor.
 */
static void fixedPointQuantization(Tensor data, Tensor res, int numCenters, float S) {
  using namespace functional;
  float multiplier = numCenters / S;

  // clip based on the scale
  Element(_1 = clip(_2, S), res, data);

  // get the quantization bin ID
  Element(_1 = round(_1 * multiplier), res);

  // revert back to floating point representation
  Element(_1 /= multiplier, res);
}

/* simulate a log-based quantization for values in data. The quantized value will be in the form of
 * S*2^q For example: 
 * data  = [0.9, 0.7, 0.5, 0.2 , 1.1] 
 * res   = [1,   0.5, 0.5, 0.25, 1  ]
 *
 * @param data contains the original data.
 * @param res will contain the resulting quantized data. set data = res for in-place operation.
 * @param size the data size.
 * @param numCenters the number of quantized centers in absolute. It should be 2^(bit-1).
 * @param S stores the scaling factor.
 * @param base for log quantized center. Default of 2.
 */
static void logQuantization(Tensor data, Tensor res, int numCenters, float S, float base = 2.0f) {
  using namespace functional;

  // clip based on the scaling factor
  Element(_1 = clip(_2, S), res, data);

  // multiplier such that the quantization is rounded in normal-space instead of log space.
  // 4/3 for base = 2. example: 11.8 should be quantized to 8, instead of 16.
  float mult = (2.0f * base) / (1.0f + base);

  // log-quantization works as the following:
  // 1. capture the sign:
  // sign = sgn(v)
  // 2. get the quantization center:
  // qc = floor(log2(abs(v/S) * _mult))
  // 3. clip the center to make sure we have no more than 2^(bit-1) centers.
  // qc = clip(qc, num_centers)
  // 4. revert back to floating point space:
  // q = 2^qc * S * sign
  //
  // The above steps are writen in 1 call as below, to avoid reserving extra Tensors:

  Element(
      _1 = sgn(_1)                                                // revert the sign back
           * S                                                    // revert the scaling function
           * pow(base,                                            // revert from log space to normal FP represtation
                 clip(floor(log(abs(_1 / S) * mult) / log(base)), // get the quantization center
                      (float) numCenters)),                       // clip
      res);
}

/* Quantize all the parameters (except bias, unless enabled via --quantize-biases).
 * Quantization only works if we store the quantization error residual.
 * The stored residual will be added for the next quantization.
 * @param graph is the model graph to be quantized (in-place).
 */
void ModelQuantizer::quantize(Ptr<ExpressionGraph> graph) {
  // lazily allocate tensor for error feedback mechanism
  if(!errorResidual_) {
    LOG(info, "Quantizing the model to {}-bits", bits_);

    int numElements = (int)graph->params()->vals()->size();
    auto allocator = New<TensorAllocator>(graph->getBackend());
    allocator->reserveExact(graph->params()->vals()->memory()->size());
    allocator->allocate(errorResidual_, {1, numElements});

    errorResidual_->set(0);

    allocators_.push_back(allocator);
    isFirstError_ = true;
  }

  {
    // apply error feedback mechanism
    using namespace functional;
    Element(_1 += _2, graph->params()->vals(), errorResidual_); // add the previous error residual to the current model 
    errorResidual_->copyFrom(graph->params()->vals()); // set the model as the error-residual (will be updated below)
  }

  for(auto p : *graph->params()) {
    // quantize weight tensors, biases optional
    if(quantBias_ || p->val()->shape()[0] > 1)
      quantizeImpl(p->val());
  }

  // get new error residual. Skip the first one.
  if (!isFirstError_) {
    using namespace functional;
    Element(_1 -= _2, errorResidual_, graph->params()->vals()); // new error-residual = original model - quantized model
  }
  else {
    errorResidual_->set(0);
    isFirstError_ = false;
  }
}


/* Tensor quantization implementation.
 * @param t is the tensor to be quantized (in-place)
 */
void ModelQuantizer::quantizeImpl(Tensor t) {
  if(!tempVar_) {
    // init the swap tensor
    auto allocator = New<TensorAllocator>(t->getBackend());
    allocator->reserveExact(sizeof(float));
    allocator->allocate(tempVar_, {1, 1});
    allocators_.push_back(allocator);
  }

  // init additional tensor for scaling optimization
  if(!delta_ && optSteps_ > 0) {
    int msize = (int) errorResidual_->size();
    auto allocator = New<TensorAllocator>(errorResidual_->getBackend());
    allocator->reserveExact(msize * sizeof(float));
    allocator->allocate(delta_, {1, msize});
    allocators_.push_back(allocator);
  }
  
  Tensor tflat = t->subtensor(0, t->size());   // flatten t for reduce

  float S = 0.0f; // scaling factor S
  // get intial scaling factor (S) based on max element in Tensor
  {
    using namespace functional;
    Reduce(abs(_1), max(_1, _2), 0.0f, tempVar_, tflat);
    S = tempVar_->get(0);
  }

  // optimize the scaling factor S
  for(int i = 0; i < optSteps_; i++) {
    Tensor q = delta_->subtensor(0, t->size());  // to store the quantized t
    
    // let t be the original tensor, and q be the quantized tensor, and q = S*a where S is the
    // scaling factor. we want to optimize S to minimize MSE(S*a - t) therefore, S =
    // sum(a*t)/sum(a*a) see https://www.aclweb.org/anthology/2020.ngt-1.4.pdf for more details.
    if(logQuant_)
      logQuantization(t, q, (1 << (bits_ - 1)) - 1, S);
    else
      fixedPointQuantization(t, q, (1 << (bits_ - 1)) - 1, S);

    // obtain a by applying q/=S
    using namespace functional;
    Element(_1 /= S, delta_);

    // get sum(a*t)
    Reduce(_1 * _2, tempVar_, tflat, q);
    float deltaNumer = tempVar_->get(0);

    // get sum(a*a)
    Reduce(_1 * _1, tempVar_, q);
    float deltaDenom = tempVar_->get(0);

    S = deltaNumer / deltaDenom;  // S = sum(a*t)/sum(a*a)
  }

  // final quantization
  if(logQuant_) {
    logQuantization(t, t, (1 << (bits_ - 1)) - 1, S);
  } else
    fixedPointQuantization(t, t,(1 << (bits_ - 1)) - 1, S);
}
}  // namespace marian
