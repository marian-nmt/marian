#pragma once

#include "marian.h"

namespace marian {

class RationalLoss {
protected:
  Expr loss_;
  Expr labels_;

  RationalLoss() = default; // protected

public:
  RationalLoss(Expr loss, Expr labels)
  : loss_(loss), labels_(labels) {}

  RationalLoss(const RationalLoss& other)
  : loss_(other.loss_), labels_(other.labels_) {}

  virtual ~RationalLoss() = default;

  Expr loss() const { return loss_; }

  template <typename T>
  void loss(std::vector<T>& losses) const {
    ABORT_IF(!loss_, "Loss has not been defined");
    loss_->val()->get(losses);
  }

  template <typename T>
  T loss() const { // this will fail if loss is not a single value
    ABORT_IF(!loss_, "Loss has not been defined");
    return loss_->val()->scalar<T>();
  }

  Expr labels() const { return labels_; }

  template <typename T>
  void labels(std::vector<T>& labels) const {
    ABORT_IF(!labels_, "Labels have not been defined");
    labels_->val()->get(labels);
  }

  template <typename T>
  T labels() const { // this will fail if loss is not a single value
    ABORT_IF(!labels_, "Labels have not been defined");
    return labels_->val()->scalar<T>();
  }

  size_t size() const {
    ABORT_IF(!labels_, "Labels have not been defined");
    return labels_->shape().elements();
  }
};

// POD for accumulating loss values after backprop
struct StaticLoss {
  float loss;
  float labels;

  StaticLoss() : loss(0.f), labels(0.f) {}

  StaticLoss(const RationalLoss& rl)
  : loss(rl.loss<float>()), labels(rl.labels<float>()) {}

  StaticLoss& operator +=(const StaticLoss& other) {
    loss = loss + other.loss;
    labels = labels + other.labels;
    return *this;
  }
};

class MultiRationalLoss : public RationalLoss {
protected:
  std::vector<RationalLoss> partialLosses_;

  virtual Expr accumulateLoss(const RationalLoss& current) = 0;
  virtual Expr accumulateLabels(const RationalLoss& current) = 0;

public:
  MultiRationalLoss() : RationalLoss() {}

  MultiRationalLoss(const RationalLoss& rl) : RationalLoss() {
    this->push_back(rl);
  }

  void push_back(const RationalLoss& current) {
    loss_   = accumulateLoss(current);
    labels_ = accumulateLabels(current);
    partialLosses_.push_back(current);
  }

  const RationalLoss& operator[](size_t i) {
    return partialLosses_[i];
  }

  auto begin() -> decltype(partialLosses_.begin()) const {
    return partialLosses_.begin();
  }

  auto end() -> decltype(partialLosses_.end()) const {
    return partialLosses_.end();
  }

  size_t size() const {
    return partialLosses_.size();
  }

};

class SumMultiRationalLoss : public MultiRationalLoss {
private:
  virtual Expr accumulateLoss(const RationalLoss& current) override {
    if(loss_)
      return loss_ + current.loss();
    else
      return current.loss();
  }

  virtual Expr accumulateLabels(const RationalLoss& current) override {
    if(labels_)
      return labels_ + current.labels();
    else
      return current.labels();
  }

public:
  SumMultiRationalLoss() : MultiRationalLoss() {}
  SumMultiRationalLoss(const RationalLoss& rl) : MultiRationalLoss(rl) {}
};

class MeanMultiRationalLoss : public MultiRationalLoss {
private:
  virtual Expr accumulateLoss(const RationalLoss& current) override {
    if(loss_)
      return loss_ + current.loss() / current.labels();
    else
      return current.loss() / current.labels();
  }

  virtual Expr accumulateLabels(const RationalLoss& current) override {
    if(labels_)
      return labels_ + 1.f; // broadcast to size
    else
      return current.labels() / current.labels(); // 1, but with correct size
  }

public:
  MeanMultiRationalLoss() : MultiRationalLoss() {}
  MeanMultiRationalLoss(const RationalLoss& rl) : MultiRationalLoss(rl) {}
};

class ScaledMultiRationalLoss : public MultiRationalLoss {
private:
  virtual Expr accumulateLoss(const RationalLoss& current) override {
    if(loss_) {
      const auto& first = partialLosses_.front();
      return loss_ + first.labels() * (current.loss() / current.labels()); // scale up/down to match scale of first loss
    } else {
      return current.loss(); // first reference loss, keeps to scale with this one
    }
  }

  virtual Expr accumulateLabels(const RationalLoss& current) override {
    if(labels_) {
      const auto& first = partialLosses_.front();
      return labels_ + first.labels() / current.labels(); // fractional label counts are OK
    } else {
      return current.labels();
    }
  }

public:
  ScaledMultiRationalLoss() : MultiRationalLoss() {}
  ScaledMultiRationalLoss(const RationalLoss& rl) : MultiRationalLoss(rl) {}
};

Ptr<MultiRationalLoss> newMultiLoss(Ptr<Options> options);

//***********************************************************************************//
// This needs some to be refactored. Currentl easiest route for backwards compat.

class LabelwiseLoss {
protected:
  std::vector<int> axes_;

  virtual Expr compute(Expr logits, Expr labelIndices,
                       Expr mask = nullptr, Expr labelWeights = nullptr) = 0;

  RationalLoss reduce(Expr loss, Expr labels) {
    ABORT_IF(!loss, "Loss has not been computed");
    ABORT_IF(!labels, "Labels have not been computed");

    Expr lossSum   = loss;
    Expr labelsSum = labels;
    for(int i = 0; i < axes_.size(); ++i) {
      lossSum   = sum(lossSum, axes_[i]);
      labelsSum = sum(labelsSum, axes_[i]);
    }

    return RationalLoss(lossSum, labelsSum);
  }

public:
  LabelwiseLoss(const std::vector<int>& axes)
  : axes_(axes) { }

  virtual RationalLoss apply(Expr logits, Expr labelIndices,
                             Expr mask = nullptr, Expr labelWeights = nullptr) {
    Expr loss = compute(logits, labelIndices, mask, labelWeights);

    Expr labels = mask ? mask                              // mask can be used as element-wise label count with broadcasting
                       : constant_like(loss, inits::ones); // we have no mask, assume all items are labels

    return reduce(loss, labels);
  }
};

class CrossEntropyLoss : public LabelwiseLoss {
public:
  CrossEntropyLoss(float smoothing)
  : LabelwiseLoss(/*axes=*/{-2, -3}), // cross-entropy already reduces over axis -1
    smoothing_(smoothing) {}

  CrossEntropyLoss(const std::vector<int>& axes, float smoothing)
  : LabelwiseLoss(axes), // cross-entropy already reduces over axis -1
    smoothing_(smoothing) {}

protected:
  float smoothing_;

  virtual Expr compute(Expr logits, Expr labelIndices,
               Expr mask = nullptr, Expr labelWeights = nullptr) override {
    Expr ce = cross_entropy(logits, labelIndices);

    if(smoothing_ > 0) {
      // @TODO: add this to CE kernels instead
      Expr ceq = mean(logsoftmax(logits), /*axis=*/ -1);
      ce = (1 - smoothing_) * ce - smoothing_ * ceq;
    }

    if(mask)
      ce = ce * mask;

    if(labelWeights)
      ce = ce * labelWeights;

    return ce;
  }
};

class RescorerLoss : public CrossEntropyLoss {
public:
  // sentence-wise CE, hence reduce only over time axis. CE reduces over last axis (-1)
  RescorerLoss() : CrossEntropyLoss(/*axes=*/{-3}, /*smoothing=*/0.f) {}

  virtual RationalLoss apply(Expr logits, Expr labelIndices,
                             Expr mask = nullptr, Expr labelWeights = nullptr) override {
    auto ce = CrossEntropyLoss::apply(logits, labelIndices, mask, labelWeights);
    return RationalLoss(-ce.loss(), ce.labels()); // we report logprobs, hence negate
  }
};

Ptr<LabelwiseLoss> newLoss(Ptr<Options> options, bool inference);

}  // namespace marian
