#include "optimizers.h"

#include "kernels/tensor_operators.h"
#include "kernels/thrust_functions.h"

namespace marian {
void Sgd::updateImpl(Tensor params, Tensor grads) {
  Element(_1 -= eta_ * _2, params, grads);
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

  Element(_1 -= (eta_ / (Sqrt(_2) + eps_)) * _3, params, gt_, grads);
}

void Adam::updateImpl(Tensor params, Tensor grads) {
  if(!mtAlloc_)
    mtAlloc_ = New<TensorAllocator>(params->getDevice());
  if(!vtAlloc_)
    vtAlloc_ = New<TensorAllocator>(params->getDevice());

  if(!mt_) {
    int elements = params->size();
    mtAlloc_->reserveExact(params->memory()->size());
    mtAlloc_->allocate(mt_, {1, elements});
    mt_->set(0);

    vtAlloc_->reserveExact(params->memory()->size());
    vtAlloc_->allocate(vt_, {1, elements});
    vt_->set(0);
  }

  t_++;
  float denom1 = 1 - std::pow(beta1_, t_);
  float denom2 = 1 - std::pow(beta2_, t_);

  Element(_1 = (beta1_ * _1) + ((1 - beta1_) * _2), mt_, grads);
  Element(_1 = (beta2_ * _1) + ((1 - beta2_) * (_2 * _2)), vt_, grads);

  Element(_1 -= eta_ * (_2 / denom1) / (Sqrt(_3 / denom2) + eps_),
          params,
          mt_,
          vt_);
}

Ptr<OptimizerBase> Optimizer(Ptr<Config> options) {
  Ptr<ClipperBase> clipper = nullptr;
  float clipNorm = options->get<double>("clip-norm");
  if(clipNorm > 0)
    clipper = Clipper<Norm>(clipNorm);

  float lrate = options->get<double>("learn-rate");

  std::string opt = options->get<std::string>("optimizer");

  if(opt == "sgd") {
    return Optimizer<Sgd>(lrate, keywords::clip = clipper);
  } else if(opt == "adagrad") {
    return Optimizer<Adagrad>(lrate, keywords::clip = clipper);
  } else if(opt == "adam") {
    return Optimizer<Adam>(lrate, keywords::clip = clipper);
  } else {
    UTIL_THROW2("Unknown optimizer: " << opt);
  }
}
}
