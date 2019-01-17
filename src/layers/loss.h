#pragma once

#include "graph/expression_operators.h"

namespace marian {

/**
 * We represent loss as pair of expressions, where loss_ is usually a sum
 * of all accumulated loss values per label and labels_ is the total number
 * of labels over which the loss was collected.
 *
 * These two values can then be used to represent various cost variants -
 * for instance label-wise cross-entropy or perplexity. Optimization is
 * only performed with regard to the summed loss_.
 *
 * Since both, loss_ and labels_ are dynamic graph nodes they can be further
 * combined into larger structures. See multi-objective losses below.
 */
class RationalLoss {
protected:
  Expr loss_;
  Expr labels_;

  RationalLoss() = default; // protected

public:
  RationalLoss(Expr loss, Expr labels)
  : loss_(loss), labels_(labels) {}

  RationalLoss(Expr loss, float labels)
  : loss_(loss),
    labels_(loss->graph()->constant({1}, inits::from_value(labels))) {}

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

/**
 * POD for accumulating loss values after forward/backward used in
 * Scheduler for updating statistics. This can only be used after a
 * successful forward step in a computation graph that owns the assigned
 * RationalLoss object.
 */
struct StaticLoss {
  float loss;
  float labels;

  StaticLoss() : loss(0.f), labels(0.f) {}

  StaticLoss(const RationalLoss& dynamic)
  : loss(dynamic.loss<float>()), labels(dynamic.labels<float>()) {}

  StaticLoss& operator +=(const StaticLoss& other) {
    loss = loss + other.loss;
    labels = labels + other.labels;
    return *this;
  }
};

/**
 * Base class for multi-objective losses which is a list of RationalLoss
 * but also defines how to accumulate that list into a single RationalLoss
 */
class MultiRationalLoss : public RationalLoss {
protected:
  std::vector<RationalLoss> partialLosses_;

  /**
   * Accumulation rule for losses
   */
  virtual Expr accumulateLoss(const RationalLoss& current) = 0;

  /**
   * Accumulation rule for labels
   */
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

/**
 * Simple sum of losses.
 * Using this makes sense when the two loss types are similar in scale and
 * number of labels. For instance two decoders over similarly sized vocabularies
 */
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

/**
 * Scaled sum of losses.
 * This can weigh losses equally by choosing the first loss_0 as a reference
 * and scaling all remaining losses loss_i by labels_0 / labels_i. Labels are
 * summed up by the same rule. By this we simulate a sum of losses at similar
 * scales. Dividing by scaled label counts yields a value close to an equally
 * weighted sum of means.
 *
 * L = sum_i^N L_i + N/M sum_j^M L_j
 *
 * We set labels to N. When reporting L/N this is equvalient to sum of means.
 * Compare to sum of means below where N is factored into the loss, but labels
 * are set to 1.
 */
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
      return labels_; // Keep first label count // or: labels_ + first.labels() / current.labels();
    } else {
      return current.labels(); // This is the first loss
    }
  }

public:
  ScaledMultiRationalLoss() : MultiRationalLoss() {}
  ScaledMultiRationalLoss(const RationalLoss& rl) : MultiRationalLoss(rl) {}
};

/**
 * Sum of mean losses.
 * Not really a rational loss as labels are factored into loss. Contribution of
 * losses is equal, same as for ScaledMultiRationalLoss, just divided by different
 * number of labels. See:
 *
 * L = (1/N sum_i^N L_i + 1/M sum_j^M L_j) = (sum_i^N L_i + N/M sum_j^M L_j) / N
 *
 * We set labels to 1. During reporting, we would see the same numbers, but gradients
 * are scaled diffrently which may result in different learning curves.
 */
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
      return labels_; // keep the existing '1'
    else
      return current.labels()->graph()->ones({1}); // just '1' as labels are factored into loss_
  }

public:
  MeanMultiRationalLoss() : MultiRationalLoss() {}
  MeanMultiRationalLoss(const RationalLoss& rl) : MultiRationalLoss(rl) {}
};

/**
 * Factory for multi-objective rational loss functions
 */
Ptr<MultiRationalLoss> newMultiLoss(Ptr<Options> options);

//***********************************************************************************//
// This needs some to be refactored. Currently easiest route for backwards compat, but
// still feels somewhat hacky.

/**
 * Computes loss per label and then reduces to RationalLoss
 */
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

/**
 * Cross entropy loss across last axis, summed up over batch and time dimensions
 */
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

/**
 * Cross entropy in rescorer used for computing sentences-level log probabilities
 */
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

/**
 * Factory for label-wise loss functions
 */
Ptr<LabelwiseLoss> newLoss(Ptr<Options> options, bool inference);

}  // namespace marian
