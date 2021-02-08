#include "optimizers.h"

#include "common/io.h"
#include "tensors/tensor_operators.h"
#include <array>

namespace marian {

float OptimizerBase::update(Tensor params, Tensor grads, size_t mbSize, float costScaleFactor) {
  int elements = (int)params->size();

  LOG_ONCE(info, "Parameter type {}, optimization type {}, casting types {}",
           params->type(), optimizerType_, castOptimizerType_);

  int numAllocateShards = 0;
  if(mvAvg_) numAllocateShards += 1; // one shard for exp smoothing
  if(castOptimizerType_) numAllocateShards += 2; // two shards for conversion

  // allocate storage for shards
  if(numAllocateShards > 0 && !baseAlloc_) {
    LOG_ONCE(info, "Allocating memory for general optimizer shards");
    baseAlloc_ = New<TensorAllocator>(params->getBackend());
    baseAlloc_->reserveExact(std::vector<size_t>(numAllocateShards, elements * sizeOf(optimizerType_)));
  }

  if(mvAvg_ && !avg_) {
    // allocate exp smooth shard tensor
    baseAlloc_->allocate(avg_, {1, elements}, optimizerType_);
    // initialize from parameters, this will be overwritten by checkpoint data if a checkpoint is found or by the first update.
    // If we resume training with no checkpoint this initialization will survive and be the basis for further averaging, which is 
    // what we want in that slightly pathological circumstance. 
    CopyCast(avg_, params);
  }

  if(castOptimizerType_) {
    if(!pm_) {
      // create parameter master copy and temporary gradient shard
      baseAlloc_->allocate(pm_, {1, elements}, optimizerType_);
      baseAlloc_->allocate(gd_, {1, elements}, optimizerType_);

      // keep parameter master copy around and initialize once, converting types
      CopyCast(pm_, params);
    }
  } else {
    // no conversion, just assign at each update
    pm_ = params;
  }

  if(!alloc_) {
    size_t size = pm_->memory()->size();
    alloc_ = New<Allocator>(pm_->getBackend()->getDeviceId(), size, size);
  }

  if(castOptimizerType_)
    CopyCast(gd_, grads);
  else
    gd_ = grads;

  // reverse cost scaling when used
  if(costScaleFactor != 1.f)
    Element(functional::_1 = functional::_1 / costScaleFactor, gd_);

  // clip gradients when used
  if(!clipper_) {
  #if 1 // @BUGBUG: when we changed to ce-sum we did not adapt gradient clipping. The norm now depends on mini-batch size, that is wrong. Keeping this for backcompat with regression tests. To be removed as soon as possible.
    float clipNorm = options_->get<float>("clip-norm", 0.f); // this is different than the dynamic scaling as it is an absolute upper limit
    if(clipNorm > 0.f) {
      clipper_ = New<NormClipper>(clipNorm);
    } else 
  #endif
    {
      clipper_ = New<ReportNormClipper>(0.f); // don't clip, just report
    }
    
    // This is a bit magical. 
    // Preallocate in order to avoid later reallocation: number of maximum GPU blocks times size of float plus some overhead.
    // This is not too critical and more an educated guess. If less memory is required we haven't lost much, if more is required
    // (unlikely) it will reallocate. The hope is to avoid GPU memory fragmentation. 
    // @TODO: check if this actually does anything beneficial, e.g. throw at reallocation and check if that ever happens.
    size_t prealloc = 65535 * 4 + 1024; 
    auto clipAlloc = New<Allocator>(pm_->getBackend()->getDeviceId(), /*bytes=*/prealloc, /*step=*/1024);
    clipper_->setAllocator(clipAlloc);
  }
  float gNorm = clipper_->clip(gd_); // clip or rescale, report norm from before clipping

  // perform update on master copy with cast gradients
  // if a type cast has been performed. Otherwise the
  // original tensors are used.
  updateImpl(pm_, gd_, mbSize);

  // if exponential smoothing is used update the average
  if(mvAvg_)
    updateAvgParams(avg_, pm_, batchesSeen_, mbSize);

  // undo paramter type cast if required
  if(castOptimizerType_)
    CopyCast(params, pm_);

  params->getBackend()->synchronize();

  return gNorm;
}

void OptimizerBase::swapWithSmoothed(Tensor params) {
  if(!mvAvg_) // no smoothing, don't do anything
    return;

  // This assumes that two swaps are going to happen eventually.
  if(castOptimizerType_) {
    // If true then optimizer type is different from the graph type,
    // hence a parameter master copy exists and we swap with the master copy.
    // We then from optimizer parameter type to graph parameter type
    pm_->swap(avg_);
    CopyCast(params, pm_);
  } else {
    // Types are equal hence there is no parameter master copy. This means
    // we need to do a proper swap between the graph params and the smoothed
    // version. We will then swap again with the next call restoring original
    // parameters. 
    params->swap(avg_);
  }
}

void OptimizerBase::load(std::vector<io::Item>& items,
                         const std::vector<Ptr<OptimizerBase>>& opts,
                         const std::vector<Ptr<Backend>>& backends,
                         const ScatterStateFunc& scatterFn,
                         bool isMainProcess) {
  isMainProcess;
  ABORT_IF(opts.size() != backends.size(), "opts and backends of different sizes??");

  size_t numShards = 0;
  if(mvAvg_) numShards += 1;
  if(castOptimizerType_) numShards += 2;

  if(castOptimizerType_) {
    io::Item iParams;
    for(auto item : items)
      if(item.name == "master_parameters")
        iParams = std::move(item);

    if(iParams.bytes.empty()) {
      LOG(warn, "[warn] Parameters not found in .npz file");
    } else {
      ABORT_IF(optimizerType_ != iParams.type,
               "Current ({}) and previous ({}) optimization type do not match",
               optimizerType_,
               iParams.type);

      scatterFn(iParams,
        [&](size_t localDeviceIndex, const char* begin, const char* end) {
          auto opt = opts[localDeviceIndex];
          if(!opt->pm_) { // lazily allocate
            size_t size = end - begin;  // this is size in bytes now
            if(!opt->baseAlloc_) {
              LOG_ONCE(info, "Allocating memory for general optimizer shards");
              opt->baseAlloc_ = New<TensorAllocator>(backends[localDeviceIndex]);
              opt->baseAlloc_->reserveExact(std::vector<size_t>(numShards, size));
            }
            int elements = (int)size / (int)sizeOf(iParams.type);
            opt->baseAlloc_->allocate(opt->pm_, {1, elements}, iParams.type);
            opt->baseAlloc_->allocate(opt->gd_, {1, elements}, iParams.type);
          }
          opt->pm_->set(begin, end, iParams.type); // set the value
        });
    }
  }

  if(mvAvg_) {
    io::Item iAvg;
    for(auto item : items)
      if(item.name == "exp_smoothing")
        iAvg = std::move(item);

    if(iAvg.bytes.empty()) {
      LOG(warn, "[warn] Average not found in .npz file");
    } else {
      ABORT_IF(optimizerType_ != iAvg.type,
          "Current ({}) and previous ({}) optimization type do not match",
          optimizerType_,
          iAvg.type);

      scatterFn(iAvg,
        [&](size_t localDeviceIndex, const char* begin, const char* end) {
          auto opt = opts[localDeviceIndex];
          if(!opt->avg_) { // lazily allocate
            size_t size = end - begin;  // this is size in bytes now
            if(!opt->baseAlloc_) {
              LOG_ONCE(info, "Allocating memory for general optimizer shards");
              opt->baseAlloc_ = New<TensorAllocator>(backends[localDeviceIndex]);
              opt->baseAlloc_->reserveExact(std::vector<size_t>(numShards, size));
            }
            int elements = (int)size / (int)sizeOf(iAvg.type);
            opt->baseAlloc_->allocate(opt->avg_, {1, elements}, iAvg.type);
          }
          opt->avg_->set(begin, end, iAvg.type); // set the value
        });
    }
  }
}

void OptimizerBase::save(std::vector<io::Item>& items,
                         const std::vector<Ptr<OptimizerBase>>& opts,
                         const GatherStateFunc& gatherFn,
                         bool isMainProcess) {
  isMainProcess;
  if(castOptimizerType_) {
    // fetch and concatenate state vectors for high precision copy
    io::Item pm = gatherFn(
      [&](size_t localDeviceIndex) {
        auto opt = opts[localDeviceIndex];
        io::Item item;
        opt->pm_->get(item, "master_parameters");
        return item;
      });
    items.emplace_back(std::move(pm));
  }
  if(mvAvg_) {
    // fetch and concatenate state vectors for smoothed parameters
    io::Item avg = gatherFn(
      [&](size_t localDeviceIndex) {
        auto opt = opts[localDeviceIndex];
        io::Item item;
        opt->avg_->get(item, "exp_smoothing");
        return item;
      });
    items.emplace_back(std::move(avg));
  }
}

void Sgd::updateImpl(Tensor params, Tensor grads, size_t actualMBSize) {
  actualMBSize; // (no correction for base update needed beyond using ce-sum)
  using namespace functional;
  Element(_1 -= eta_ * _2, 
          params, 
          grads);
}

void Sgd::load(std::vector<io::Item>& items,
               const std::vector<Ptr<OptimizerBase>>& opts,
               const std::vector<Ptr<Backend>>& backends,
               const ScatterStateFunc& scatterFn,
               bool isMainProcess) {
  OptimizerBase::load(items, opts, backends, scatterFn, isMainProcess);
}

void Sgd::save(std::vector<io::Item>& items,
               const std::vector<Ptr<OptimizerBase>>& opts,
               const GatherStateFunc& gatherFn,
               bool isMainProcess) {
  OptimizerBase::save(items, opts, gatherFn, isMainProcess); // collect parameters from base
}


// Adagrad
void Adagrad::updateImpl(Tensor params, Tensor grads, size_t actualMBSize) {
  actualMBSize; // not used in Adagrad

  // allocate optimizer-specific parameters
  if(!alloc_) {
    LOG_ONCE(info, "Allocating memory for Adagrad-specific shards");
    alloc_ = New<TensorAllocator>(params->getBackend());
  }

  if(!gt_) {
    int elements = (int)params->size();
    alloc_->reserveExact(params->memory()->size());
    alloc_->allocate(gt_, {1, elements}, params->type());
    gt_->set(0.f);
  }

  using namespace functional;

  Element(_1 += (_2 * _2), gt_, grads);

  // make sure eps_ does not drop below smallest (positive) value, add some reserve by multiplying with 2
  eps_ = (float)std::max(NumericLimits<double>(params->type()).min * 2.f, (double)eps_);
  Element(_1 -= (eta_ / (sqrt(_2) + eps_)) * _3, 
          params, 
          gt_, 
          grads);
}

void Adagrad::load(std::vector<io::Item>& items,
                   const std::vector<Ptr<OptimizerBase>>& opts,
                   const std::vector<Ptr<Backend>>& backends,
                   const ScatterStateFunc& scatterFn,
                   bool isMainProcess) {
  OptimizerBase::load(items, opts, backends, scatterFn, isMainProcess);

  if(isMainProcess)
    LOG(info, "Loading Adagrad parameters");

  io::Item iGt;
  for(auto item : items)
    // extract data into vectors
    if(item.name == "adagrad_gt")
      iGt = std::move(item);

  if(iGt.bytes.empty()) {
    LOG(warn, "[warn] Adagrad parameters not found in checkpoint");
    return;
  }

  ABORT_IF(optimizerType_ != iGt.type,
          "Current ({}) and previous ({}) optimization type do not match",
          optimizerType_,
          iGt.type);

  scatterFn(iGt,
    [&](size_t localDeviceIndex, const char* begin, const char* end) {
      auto opt = std::dynamic_pointer_cast<Adagrad>(opts[localDeviceIndex]);
      if(!opt->gt_) {
        if(!opt->alloc_)
          opt->alloc_ = New<TensorAllocator>(backends[localDeviceIndex]);

        size_t size = end - begin; // this is size in bytes now
        int elements = (int)size / (int)sizeOf(iGt.type);
        opt->alloc_->reserveExact(size);
        opt->alloc_->allocate(opt->gt_, {1, elements}, iGt.type);
      }

      opt->gt_->set(begin, end, iGt.type);
    });
}

void Adagrad::save(std::vector<io::Item>& items,
                   const std::vector<Ptr<OptimizerBase>>& opts,
                   const GatherStateFunc& gatherFn,
                   bool isMainProcess) {
  OptimizerBase::save(items, opts, gatherFn, isMainProcess); // collect parameters from base

  if(isMainProcess)
    LOG(info, "Saving Adagrad parameters");

  // fetch and concatenate state vectors from distributed shards into a CPU-side vector
  io::Item gt = gatherFn(
    [&](size_t localDeviceIndex) {
      auto opt = std::dynamic_pointer_cast<Adagrad>(opts[localDeviceIndex]);
      io::Item item;
      opt->gt_->get(item, "adagrad_gt");
      return item;
    });
  items.emplace_back(std::move(gt));
}

void Adagrad::resetStats() {
  if(gt_)
    gt_->set(0.f);
}

// Adam
void Adam::updateImpl(Tensor params, Tensor grads, size_t actualMBSize) {
  // lazy allocation
  if(!alloc_) {
    LOG_ONCE(info, "Allocating memory for Adam-specific shards");
    alloc_ = New<TensorAllocator>(params->getBackend());
  }

  if(!mt_) {
    int elements = (int)params->size();
    size_t shard = (size_t)elements * sizeOf(params->type());
    alloc_->reserveExact({shard, shard});

    alloc_->allocate(mt_, {1, elements}, params->type());
    mt_->set(0.f);

    alloc_->allocate(vt_, {1, elements}, params->type());
    vt_->set(0.f);
  }

  double T = 1, Tref = 1;
  if(OptimizerBase::refMBWordsParam_ > 0) {
    T = (double)actualMBSize;
    if(actualMBSize > refBatchTrgWords_)
      Tref = (double)refMBWordsParam_;
    else 
      Tref = T;
  }

  // adjust for minibatch-size changes if Adam parameters are given a reference size (else do nothing)
  // Why the T/Tref factor on eta? The Adam optimizer adds an RMS-normalized gradient
  // value (times learning rate) to the model. We know that for Tref, that learning rate is good.
  // If we increase the batch size by (T/Tref), then without adjustment, we would still add an
  // RMS-normalized gradient value. That means that the contribution of an individual label is
  // now weighted down by (T/Tref). However, batch-size agnostic hyper-parameterization aims to keep
  // the weight on the contribution of each label gradient invariant. Thus, we must undo that
  // down-weighting, by multiplying the RMS-normalized gradient value by an additional factor
  // of (T/Tref). This is implemented here by locally multiplying the learning rate
  // with that factor.
  double eta   = eta_ * (T / Tref);
  double beta1 = beta1_;
  double beta2 = beta2_;
  double decay = w_    ;

  // denominators. At steady state: =1. This recursion does the same as the Adam beta correction term.
  denom1_ = (beta1 * denom1_) + (1 - beta1); // momentum smoothing
  denom2_ = (beta2 * denom2_) + (1 - beta2); // RMS normalization

  // numerators. Divide by T to convert ce-sum gradient to avg gradient.
  using namespace functional;
#if 0 // why the division by T or T^2 here? It's T=1 without mb-ref anyway and we have the adjustment above, also converges a lot(!) slower with T != 1
  Element(_1 = ((float)beta1 * _1) + float((1 - beta1) / T    ) *  _2,       mt_, grads); // momentum smoothing. At steady state: =smoothed avg gradient
  Element(_1 = ((float)beta2 * _1) + float((1 - beta2) / T / T) * (_2 * _2), vt_, grads); // RMS normalization.  At steady state: =mean square of the avg gradients
#else
  Element(_1 = ((float)beta1 * _1) + float((1 - beta1)) *  _2,       mt_, grads); // momentum smoothing. At steady state: =smoothed avg gradient
  Element(_1 = ((float)beta2 * _1) + float((1 - beta2)) * (_2 * _2), vt_, grads); // RMS normalization.  At steady state: =mean square of the avg gradients
#endif

  // make sure eps_ does not drop below minimum value, this is important
  // when training with mixed precision. Otherwise we divide by 0.
  // We multiply the minimum by 2 in order to step away from the abyss.
  eps_ = std::max(NumericLimits<float>(params->type()).min * 2.f, eps_);

  // make sure eps_ does not drop below minimum value, this is important
  // when training with mixed precision. Otherwise we divide by 0.
  // We multiply the minimum by 2 in order to step away from the abyss.
  eps_ = std::max(NumericLimits<float>(params->type()).min * 2.f, eps_);

  // apply Adam normalization
  float etaf = (float)eta, denom1f = (float)denom1_, denom2f = (float)denom2_, decayf = (float)decay; // (get casts out of Element expression for readability)
  Element(_1 -= etaf                               // learning-rate: x_t = x_{t-1} - \eta * (...)
                * ((  (     _2 / denom1f)          // momentum-smoothed per-sample gradient: m_{t-1}
                    / (sqrt(_3 / denom2f) + eps_)) // normalize by RMS: \sqrt(v_{t-1})
                   + (decayf * _1)),                 // weight-decay: w * x_{t-1}
          params,  // =_1
          mt_,     // =_2
          vt_      // =_3
          );
}

void Adam::load(std::vector<io::Item>& items,
                const std::vector<Ptr<OptimizerBase>>& opts,
                const std::vector<Ptr<Backend>>& backends,
                const ScatterStateFunc& scatterFn,
                bool isMainProcess) {
  OptimizerBase::load(items, opts, backends, scatterFn, isMainProcess);

  if(isMainProcess)
    LOG(info, "Loading Adam parameters");

  io::Item iMt;
  io::Item iVt;
  std::array<double, 2> vDenoms;

  for(auto item : items) {
    // extract data into vectors
    if(item.name == "adam_mt") {
      iMt = std::move(item);
    } else if(item.name == "adam_vt") {
      iVt = std::move(item);
    } else if(item.name == "adam_denoms") {
      ABORT_IF(item.size() != 2 * sizeof(double), "adam_denoms should have 2 entries not {} bytes", item.size());
      std::copy((double*)item.data(), ((double*)item.data()) + 2, vDenoms.begin());
      // Back compat note: Old files lacked "adam_denoms". For those, vDenoms will remain 0, which reproduces the old behavior.
    }
  }

  if(iMt.bytes.empty() || iVt.bytes.empty()) {
    LOG(warn, "[warn] Adam parameters not found in .npz file");
    return;
  }

  ABORT_IF(optimizerType_ != iMt.type,
           "Current ({}) and previous ({}) optimization type do not match",
           optimizerType_,
           iMt.type);

  ABORT_IF(iMt.size() != iVt.size(), "mt and vt have different sizes??");

  scatterFn(iMt,
    [&](size_t localDeviceIndex, const char* begin, const char* end) {
      auto opt = std::dynamic_pointer_cast<Adam>(opts[localDeviceIndex]);

      // denominators need to be set in all shards, hijack this scatter
      opt->denom1_ = vDenoms[0];
      opt->denom2_ = vDenoms[1];

      if(!opt->mt_ || !opt->vt_) { // lazily allocate
        if(!opt->alloc_)
          opt->alloc_ = New<TensorAllocator>(backends[localDeviceIndex]);
        size_t size = end - begin;  // this is size in bytes now
        int elements = (int)size / (int)sizeOf(iMt.type);
        opt->alloc_->reserveExact(2 * size);
        opt->alloc_->allocate(opt->mt_, {1, elements}, iMt.type);
        opt->alloc_->allocate(opt->vt_, {1, elements}, iMt.type);
      }
      opt->mt_->set(begin, end, iMt.type); // set the value
    });

  scatterFn(iVt,
    [&](size_t localDeviceIndex, const char* begin, const char* end) {
      auto opt = std::dynamic_pointer_cast<Adam>(opts[localDeviceIndex]);
      opt->vt_->set(begin, end, iVt.type);
    });
}

void Adam::save(std::vector<io::Item>& items,
                const std::vector<Ptr<OptimizerBase>>& opts,
                const GatherStateFunc& gatherFn,
                bool isMainProcess) {
  OptimizerBase::save(items, opts, gatherFn, isMainProcess); // collect parameters from base

  if(isMainProcess)
    LOG(info, "Saving Adam parameters");

  // fetch and concatenate state vectors from distributed shards into a CPU-side vector
  io::Item mt = gatherFn(
    [&](size_t localDeviceIndex) {
      auto opt = std::dynamic_pointer_cast<Adam>(opts[localDeviceIndex]);
      io::Item item;
      opt->mt_->get(item, "adam_mt");
      return item;
    });
  items.emplace_back(std::move(mt));

  io::Item vt = gatherFn(
    [&](size_t localDeviceIndex) {
      auto opt = std::dynamic_pointer_cast<Adam>(opts[localDeviceIndex]);
      io::Item item;
      opt->vt_->get(item, "adam_vt");
      return item;
    });
  items.emplace_back(std::move(vt));

  std::vector<double> vDenoms{denom1_, denom2_};
  items.emplace_back(io::fromVector(vDenoms, "adam_denoms"));
}

void Adam::resetStats() {
  if(mt_)
    mt_->set(0.f);

  if(vt_)
    vt_->set(0.f);

  denom1_ = 0; // @BUGBUG: or 1 or refMBWords if so specified. Fix once we have proper parameterization for that.
  denom2_ = 0;
}

Ptr<OptimizerBase> Optimizer(Ptr<Options> options) {
  auto optType = options->get<std::string>("optimizer");
  auto params = options->has("optimizer-params")
                     ? options->get<std::vector<float>>("optimizer-params")
                     : std::vector<float>({});
  Ptr<OptimizerBase> opt;
  if(optType == "sgd") {
    opt = New<Sgd>(options);
  } else if(optType == "adagrad") {
    opt = New<Adagrad>(options);
  } else if(optType == "adam") {
    opt = New<Adam>(options);
  } else {
    ABORT("Unknown optimizer type: {}", optType);
  }

  opt->setParams(params);
  return opt;
}

}  // namespace marian
