#include "optimizers.h"

#include "common/io.h"
#include "tensors/tensor_operators.h"

namespace marian {

void Sgd::updateImpl(Tensor params, Tensor grads) {
  using namespace functional;
  Element(_1 -= eta_ * _2,
          params,
          grads);

  params->getBackend()->synchronize();
}

// Aagrad

void Adagrad::updateImpl(Tensor params, Tensor grads) {
  if(!alloc_)
    alloc_ = New<TensorAllocator>(params->getBackend());

  if(!gt_) {
    int elements = (int)params->size();
    alloc_->reserveExact(params->memory()->size());
    alloc_->allocate(gt_, {1, elements});
    gt_->set(0.f);
  }

  using namespace functional;

  Element(_1 += (_2 * _2), gt_, grads);

  Element(_1 -= (eta_ / (sqrt(_2) + eps_)) * _3,
          params,
          gt_,
          grads);

  params->getBackend()->synchronize();
}

void Adagrad::load(const std::string& name,
                   const std::vector<Ptr<OptimizerBase>>& opts,
                   const std::vector<Ptr<Backend>>& backends,
                   const ScatterStateFunc& scatterFn) {
  ABORT_IF(opts.size() != backends.size(), "opts and backends of different sizes??");

  if(!filesystem::exists(name))
    return;

  LOG(info, "Loading Adagrad parameters from {}", name);

  std::vector<float> vGt;

  // @TODO: use new IO
  auto items = io::loadItems(name);
  for(auto item : items) {
    // get the size of gt_
    auto totalSize = item.shape.elements();

    // extract data into vectors
    if(item.name == "adagrad_gt") {
      vGt.resize(totalSize);
      std::copy(
          (float*)item.data(), (float*)item.data() + totalSize, vGt.begin());
    }
  }
  if(vGt.empty()) {
    LOG(warn, "[warn] Adagrad parameters not found in .npz file");
    return;
  }

  scatterFn(vGt,
    [&](size_t localDeviceIndex, std::vector<float>::const_iterator begin, std::vector<float>::const_iterator end) {
    auto opt = std::dynamic_pointer_cast<Adagrad>(opts[localDeviceIndex]);
    if(!opt->gt_) {
      if(!opt->alloc_)
        opt->alloc_ = New<TensorAllocator>(backends[localDeviceIndex]);
      auto size = end-begin;
      opt->alloc_->reserveExact(sizeof(float) * size);
      opt->alloc_->allocate(opt->gt_, {1, (int)size});
    }
    opt->gt_->set(std::vector<float>(begin, end));
  });
}

void Adagrad::save(const std::string& name,
                   const std::vector<Ptr<OptimizerBase>>& opts,
                   const GatherStateFunc& gatherFn,
                   bool isMainProcess /*= true*/) {
  LOG(info, "Saving Adagrad parameters to {}", name);

  // fetch and concatenate state vectors from distributed shards into a CPU-side vector
  auto vGt = gatherFn([&](size_t localDeviceIndex) {
      auto opt = std::dynamic_pointer_cast<Adagrad>(opts[localDeviceIndex]);
      std::vector<float> data;
      opt->gt_->get(data);
      return data;
    });

  // if not main MPI process then we have done our duty
  if (!isMainProcess)
    return;

  // save to file
  io::Item item;
  item.name = "adagrad_gt";
  item.shape = Shape({1, (int)vGt.size()});
  item.type = Type::float32;
  item.bytes.resize(vGt.size() * sizeOf(item.type));
  std::copy(
      (char*)vGt.data(), (char*)vGt.data() + vGt.size(), item.bytes.begin());

  io::saveItems(name, {item});
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
    int elements = (int)params->size();
    alloc_->reserveExact(2 * params->memory()->size());
    alloc_->allocate(mt_, {1, elements});
    mt_->set(0.f);

    alloc_->allocate(vt_, {1, elements});
    vt_->set(0.f);
  }

  t_++;
  float denom1 = 1 - (float)std::pow(beta1_, t_);
  float denom2 = 1 - (float)std::pow(beta2_, t_);

  using namespace functional;

  Element(_1 = (beta1_ * _1) + ((1 - beta1_) * _2), mt_, grads);
  Element(_1 = (beta2_ * _1) + ((1 - beta2_) * (_2 * _2)), vt_, grads);

  Element(_1 -= eta_                         // learning-rate: x_t = x_{t-1} - \eta * (...)
                * ((_2 / denom1)             // 1st moment: m_{t-1}
                / (sqrt(_3 / denom2) + eps_) // 2nd moment: \sqrt(v_{t-1})
                + w_ * _1),                  // weight-decay: w * x_{t-1}
          params,
          mt_,
          vt_);

  params->getBackend()->synchronize();
}

void Adam::load(const std::string& name,
                const std::vector<Ptr<OptimizerBase>>& opts,
                const std::vector<Ptr<Backend>>& backends,
                const ScatterStateFunc& scatterFn) {
  ABORT_IF(opts.size() != backends.size(), "opts and backends of different sizes??");

  if(!filesystem::exists(name))
    return;

  LOG(info, "Loading Adam parameters from {}", name);

  std::vector<float> vMt;
  std::vector<float> vVt;

  auto items = io::loadItems(name);
  for(auto item : items) {
    // get the size of mt_ and vt_, they are the same
    auto totalSize = item.shape.elements();

    // extract data into vectors
    if(item.name == "adam_mt") {
      vMt.resize(totalSize);
      std::copy(
          (float*)item.data(), (float*)item.data() + totalSize, vMt.begin());
    }
    if(item.name == "adam_vt") {
      vVt.resize(totalSize);
      std::copy(
          (float*)item.data(), (float*)item.data() + totalSize, vVt.begin());
    }
  }
  if(vMt.empty() || vVt.empty()) {
    LOG(warn, "[warn] Adam parameters not found in .npz file");
    return;
  }
  ABORT_IF(vMt.size() != vVt.size(), "mt and vt have different sizes??");

  //LOG(info, "loading Adam params");
  scatterFn(vMt,
    [&](size_t localDeviceIndex, std::vector<float>::const_iterator begin, std::vector<float>::const_iterator end) {
    auto opt = std::dynamic_pointer_cast<Adam>(opts[localDeviceIndex]);
    if(!opt->mt_ || !opt->vt_) { // lazily allocate
      if(!opt->alloc_)
        opt->alloc_ = New<TensorAllocator>(backends[localDeviceIndex]);
      auto size = end-begin;
      opt->alloc_->reserveExact(2 * sizeof(float) * size);
      opt->alloc_->allocate(opt->mt_, {1, (int)size});
      opt->alloc_->allocate(opt->vt_, {1, (int)size});
    }
    opt->mt_->set(std::vector<float>(begin, end)); // set the value
  });

  scatterFn(vVt,
    [&](size_t id, std::vector<float>::const_iterator begin, std::vector<float>::const_iterator end) {
    auto opt = std::dynamic_pointer_cast<Adam>(opts[id]);
    opt->vt_->set(std::vector<float>(begin, end));
  });
  //LOG(info, "done loading Adam params");
}

void Adam::save(const std::string& name,
                const std::vector<Ptr<OptimizerBase>>& opts,
                const GatherStateFunc& gatherFn,
                bool isMainProcess /*= true*/) {
  if (isMainProcess)
    LOG(info, "Saving Adam parameters to {}", name);

  // fetch and concatenate state vectors from distributed shards into a CPU-side vector
  auto vMt = gatherFn([&](size_t localDeviceIndex) {
    auto opt = std::dynamic_pointer_cast<Adam>(opts[localDeviceIndex]);
    std::vector<float> data;
    opt->mt_->get(data);
    return data;
  });

  auto vVt = gatherFn([&](size_t localDeviceIndex) {
    auto opt = std::dynamic_pointer_cast<Adam>(opts[localDeviceIndex]);
    std::vector<float> data;
    opt->vt_->get(data);
    return data;
  });

  // if not main MPI process then we have done our duty
  if (!isMainProcess)
      return;

  // save to file
  io::Item itemMt;
  itemMt.name = "adam_mt";
  itemMt.shape = Shape({1, (int)vMt.size()});
  itemMt.type = Type::float32;
  itemMt.bytes.resize(vMt.size() * sizeOf(itemMt.type));
  std::copy(
      (char*)vMt.data(), (char*)vMt.data() + vMt.size(), itemMt.bytes.begin());

  io::Item itemVt;
  itemVt.name = "adam_vt";
  itemVt.shape = Shape({1, (int)vVt.size()});
  itemVt.type = Type::float32;
  itemVt.bytes.resize(vVt.size() * sizeOf(itemVt.type));
  std::copy(
      (char*)vVt.data(), (char*)vVt.data() + vVt.size(), itemVt.bytes.begin());

  io::saveItems(name, {itemMt, itemVt});
}

void Adam::resetStats() {
  if(mt_)
    mt_->set(0.f);

  if(vt_)
    vt_->set(0.f);
}

Ptr<OptimizerBase> Optimizer(Ptr<Options> options) {
  float lrate = (float)options->get<double>("learn-rate"); // @TODO: should this be <float>?
  auto params = options->has("optimizer-params")
                    ? options->get<std::vector<float>>("optimizer-params")
                    : std::vector<float>({});

  Ptr<ClipperBase> clipper = nullptr;
  float clipNorm = (float)options->get<double>("clip-norm"); // @TODO: should this be <float>?
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
}  // namespace marian
