#include "optimizers.h"

#include "tensors/tensor_operators.h"

namespace marian {

void Sgd::updateImpl(Tensor params, Tensor grads) {
  using namespace functional;
  Element(_1 -= (multiplyFactor_ * eta_) * _2, params, grads);

  params->getBackend()->synchronize();
}

// Aagrad

void Adagrad::updateImpl(Tensor params, Tensor grads) {
  if(!alloc_)
    alloc_ = New<TensorAllocator>(params->getBackend());

  if(!gt_) {
    int elements = params->size();
    alloc_->reserveExact(params->memory()->size());
    alloc_->allocate(gt_, {1, elements});
    gt_->set(0.f);
  }

  using namespace functional;

  Element(_1 += (_2 * _2), gt_, grads);

  Element(_1 -= ((multiplyFactor_ * eta_) / (sqrt(_2) + eps_)) * _3,
          params,
          gt_,
          grads);

  params->getBackend()->synchronize();
}

void Adagrad::load(const std::string& name,
                   std::vector<Ptr<OptimizerBase>> opts,
                   std::vector<Ptr<Backend>> backends) {
  if(!boost::filesystem::exists(name))
    return;

  LOG(info, "Loading Adagrad parameters from {}", name);

  std::vector<float> vGt;
  size_t totalSize = 0;

  auto numpy = cnpy::npz_load(name);
  for(auto it : numpy) {
    auto name = it.first;
    auto np = it.second;

    // get the size of gt_
    totalSize = np->shape[1];

    // extract data into vectors
    if(name == "adagrad_gt") {
      vGt.resize(totalSize);
      std::copy(
          (float*)np->data(), (float*)np->data() + totalSize, vGt.begin());
    }
  }

  if(vGt.empty()) {
    LOG(warn, "[warn] Adagrad parameters not found in .npz file");
    return;
  }

  // get the size of params which should go
  size_t shardSize = ceil(totalSize / (float)backends.size());

  size_t id = 0;
  for(auto optBase : opts) {
    auto opt = std::dynamic_pointer_cast<Adagrad>(optBase);

    int size = std::min(shardSize, totalSize);
    totalSize -= size;

    if(!opt->alloc_)
      opt->alloc_ = New<TensorAllocator>(backends[id]);

    if(!opt->gt_) {
      opt->alloc_->reserveExact(sizeof(float) * size);
      opt->alloc_->allocate(opt->gt_, {1, size});
    }

    size_t shift = id * shardSize;
    std::vector<float> tmp(vGt.begin() + shift, vGt.begin() + shift + size);
    opt->gt_->set(tmp);

    id++;
  }
}

void Adagrad::save(const std::string& name,
                   std::vector<Ptr<OptimizerBase>> opts,
                   size_t totalSize) {
  LOG(info, "Saving Adagrad parameters to {}", name);

  std::vector<float> vGt;

  for(auto optBase : opts) {
    auto opt = std::dynamic_pointer_cast<Adagrad>(optBase);
    std::vector<float> tmp;
    opt->gt_->get(tmp);
    vGt.insert(vGt.end(), tmp.begin(), tmp.end());
  }

  unsigned int shape[2] = { 1, (unsigned int)vGt.size() };

  cnpy::npz_save(name, "adagrad_gt", vGt.data(), shape, 2, "w");
}

void Adagrad::resetStats() {
  if(gt_)
    gt_->set(0.f);
}

// Adam

void Adam::updateImpl(Tensor params, Tensor grads) {
  if(!alloc_)
    alloc_ = New<TensorAllocator>(params->getBackend());

  if(!mt_) {
    int elements = params->size();
    alloc_->reserveExact(2 * params->memory()->size());
    alloc_->allocate(mt_, {1, elements});
    mt_->set(0.f);

    alloc_->allocate(vt_, {1, elements});
    vt_->set(0.f);
  }

  t_++;
  float denom1 = 1 - std::pow(beta1_, t_);
  float denom2 = 1 - std::pow(beta2_, t_);

  using namespace functional;

  Element(_1 = (beta1_ * _1) + ((1 - beta1_) * _2), mt_, grads);
  Element(_1 = (beta2_ * _1) + ((1 - beta2_) * (_2 * _2)), vt_, grads);

  Element(_1 -= (multiplyFactor_ * eta_) * (_2 / denom1)
                / (sqrt(_3 / denom2) + eps_),
          params,
          mt_,
          vt_);

  params->getBackend()->synchronize();
}

void Adam::load(const std::string& name,
                std::vector<Ptr<OptimizerBase>> opts,
                std::vector<Ptr<Backend>> backends) {
  if(!boost::filesystem::exists(name))
    return;

  LOG(info, "Loading Adam parameters from {}", name);

  std::vector<float> vMt;
  std::vector<float> vVt;
  size_t totalSize = 0;

  auto numpy = cnpy::npz_load(name);
  for(auto it : numpy) {
    auto name = it.first;
    auto np = it.second;

    // get the size of mt_ and vt_, they are the same
    totalSize = np->shape[1];

    // extract data into vectors
    if(name == "adam_mt") {
      vMt.resize(totalSize);
      std::copy(
          (float*)np->data(), (float*)np->data() + totalSize, vMt.begin());
    }
    if(name == "adam_vt") {
      vVt.resize(totalSize);
      std::copy(
          (float*)np->data(), (float*)np->data() + totalSize, vVt.begin());
    }
  }

  if(vMt.empty() || vVt.empty()) {
    LOG(warn, "[warn] Adam parameters not found in .npz file");
    return;
  }

  // get the size of params which should go
  size_t shardSize = ceil(totalSize / (float)backends.size());

  size_t id = 0;
  for(auto optBase : opts) {
    auto opt = std::dynamic_pointer_cast<Adam>(optBase);

    int size = std::min(shardSize, totalSize);
    totalSize -= size;

    if(!opt->alloc_)
      opt->alloc_ = New<TensorAllocator>(backends[id]);

    if(!opt->mt_ || !opt->vt_) {
      opt->alloc_->reserveExact(2 * sizeof(float) * size);
      opt->alloc_->allocate(opt->mt_, {1, size});
      opt->alloc_->allocate(opt->vt_, {1, size});
    }

    size_t shift = id * shardSize;
    std::vector<float> tmpMt(vMt.begin() + shift, vMt.begin() + shift + size);
    opt->mt_->set(tmpMt);
    std::vector<float> tmpVt(vVt.begin() + shift, vVt.begin() + shift + size);
    opt->vt_->set(tmpVt);

    id++;
  }
}

void Adam::save(const std::string& name,
                std::vector<Ptr<OptimizerBase>> opts,
                size_t totalSize) {
  LOG(info, "Saving Adam parameters to {}", name);

  std::vector<float> vMt;
  std::vector<float> vVt;

  for(auto optBase : opts) {
    auto opt = std::dynamic_pointer_cast<Adam>(optBase);

    std::vector<float> tmp;
    opt->mt_->get(tmp);
    vMt.insert(vMt.end(), tmp.begin(), tmp.end());
    opt->vt_->get(tmp);
    vVt.insert(vVt.end(), tmp.begin(), tmp.end());
  }

  // the shape is the same for mt_ and vt_
  std::vector<unsigned int> shape{ 1, (unsigned int)vMt.size() };

  cnpy::npz_save(name,
                     {
                       cnpy::NpzItem("adam_mt", vMt, shape),
                       cnpy::NpzItem("adam_vt", vVt, shape)
                     });
}

void Adam::resetStats() {
  if(mt_)
    mt_->set(0.f);

  if(vt_)
    vt_->set(0.f);
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
    ABORT("Unknown optimizer: {}", opt);
  }
}
}
