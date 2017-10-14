#include "optimizers.h"

#include "kernels/tensor_operators.h"
#include "kernels/thrust_functions.h"

namespace marian {
void Sgd::updateImpl(Tensor params, Tensor grads) {
  Element(_1 -= (multiplyFactor_ * eta_) * _2, params, grads);

  cudaStreamSynchronize(0);
}

void Adagrad::updateImpl(Tensor params, Tensor grads) {
  if(!alloc_)
    alloc_ = New<TensorAllocator>(params->getDevice());

  if(!gt_) {
    int elements = params->size();
    alloc_->reserveExact(params->memory()->size());
    alloc_->allocate(gt_, {1, elements});
    gt_->set(0);
  }

  Element(_1 += (_2 * _2), gt_, grads);

  Element(_1 -= ((multiplyFactor_ * eta_) / (Sqrt(_2) + eps_)) * _3,
          params,
          gt_,
          grads);

  cudaStreamSynchronize(0);
}

void Adagrad::resetStats() {
  if(gt_)
    gt_->set(0);
  cudaStreamSynchronize(0);
}

void Adam::updateImpl(Tensor params, Tensor grads) {
  if(!alloc_)
    alloc_ = New<TensorAllocator>(params->getDevice());

  if(!mt_) {
    int elements = params->size();
    alloc_->reserveExact(2 * params->memory()->size());
    alloc_->allocate(mt_, {1, elements});
    mt_->set(0);

    alloc_->allocate(vt_, {1, elements});
    vt_->set(0);
  }

  t_++;
  float denom1 = 1 - std::pow(beta1_, t_);
  float denom2 = 1 - std::pow(beta2_, t_);

  Element(_1 = (beta1_ * _1) + ((1 - beta1_) * _2), mt_, grads);
  Element(_1 = (beta2_ * _1) + ((1 - beta2_) * (_2 * _2)), vt_, grads);

  Element(_1 -= (multiplyFactor_ * eta_) * (_2 / denom1)
                / (Sqrt(_3 / denom2) + eps_),
          params,
          mt_,
          vt_);

  cudaStreamSynchronize(0);
}

void Adam::resetStats() {
  if(mt_)
    mt_->set(0);

  if(vt_)
    vt_->set(0);

  cudaStreamSynchronize(0);
}

Ptr<OptimizerBase> Optimizer(Ptr<Config> options) {
  float lrate = options->get<double>("learn-rate");
  auto params = options->has("optimizer-params")
                    ? options->get<std::vector<float>>("optimizer-params")
                    : std::vector<float>({});

  Ptr<ClipperBase> clipper = nullptr;
  float clipNorm = options->get<double>("clip-norm");
  if(clipNorm > 0)
    clipper = Clipper<Norm>(clipNorm);

  auto opt = options->get<std::string>("optimizer");

  if(opt == "sgd") {
    return Optimizer<Sgd>(lrate, clipper, params);
  } else if(opt == "adagrad") {
    return Optimizer<Adagrad>(lrate, clipper, params);
  } else if(opt == "adam") {
    return Optimizer<Adam>(lrate, clipper, params);
  } else {
    UTIL_THROW2("Unknown optimizer: " << opt);
  }
}
}
