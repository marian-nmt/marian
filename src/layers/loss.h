#pragma once

#include "graph/expression_operators.h"

namespace marian {

/**
 * We represent loss as pair of expressions, where loss_ is usually a sum
 * of all accumulated loss values per label and count_ is the total number
 * of labels over which the loss was collected.
 *
 * These two values can then be used to represent various cost variants -
 * for instance label-wise cross-entropy or perplexity. Optimization is
 * only performed with regard to the summed loss_.
 *
 * Since both, loss_ and count_ are dynamic graph nodes they can be further
 * combined into larger structures. See multi-objective losses below.
 */
class RationalLoss {
protected:
  Expr loss_;  // numerator
  Expr count_; // denominator

  RationalLoss() = default; // protected

public:
  RationalLoss(Expr loss, Expr count)
  : loss_(loss), count_(count) {}

  RationalLoss(Expr loss, float count)
  : loss_(loss),
    count_(constant_like(loss, inits::from_value(count))) {}

  RationalLoss(const RationalLoss& other)
  : loss_(other.loss_), count_(other.count_) {}

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

  Expr count() const { return count_; }

  template <typename T>
  void count(std::vector<T>& labels) const {
    ABORT_IF(!count_, "Labels have not been defined");
    count_->val()->get(labels);
  }

  template <typename T>
  T count() const { // this will fail if loss is not a single value
    ABORT_IF(!count_, "Labels have not been defined");
    return count_->val()->scalar<T>();
  }

  // @TODO: add a funtion for returning maybe ratio?

  size_t size() const {
    ABORT_IF(!count_, "Labels have not been defined");
    return count_->shape().elements();
  }
};

/**
 * POD for accumulating loss values after forward/backward used in
 * Scheduler for updating statistics. This can only be used after a
 * successful forward step in a computation graph that owns the assigned
 * RationalLoss object.
 */
struct StaticLoss {
  float loss;  // numerator
  float count; // denominator

  StaticLoss() : loss(0.f), count(0.f) {}

  StaticLoss(const RationalLoss& dynamic)
  : loss(dynamic.loss<float>()), count(dynamic.count<float>()) {}

  StaticLoss& operator +=(const StaticLoss& other) {
    loss = loss + other.loss;
    count = count + other.count;
    return *this;
  }

  void reset() {
    loss = 0.f;
    count = 0.f;
  }
};

/**
 * @brief Base class for multi-objective losses
 * Base class for multi-objective losses which is a list of RationalLoss
 * but also defines how to accumulate that list into a single RationalLoss
 */
class MultiRationalLoss : public RationalLoss {
protected:
  std::vector<RationalLoss> partialLosses_;

  /**
   * @brief Accumulation rule for losses
   * In the default case this would just be a sum, see SumMultiRationalLoss, but there are
   * special cases like ScaledMultiRationalLoss (scale other loses according to first label count)
   * or MeanMultiRationalLoss (sum of means) where the accumulation is more complex.
   */
  virtual Expr accumulateLoss(const RationalLoss& current) = 0;

  /**
   * @brief Accumulation rule for labels
   * Similar as above, the naive case is summation, but for instance MeanMultiRationalLoss
   * is including all label counts in the loss hence label counts are always just 1 which is
   * passed through without summation or other modifications.
   */
  virtual Expr accumulateCount(const RationalLoss& current) = 0;

public:
  MultiRationalLoss() : RationalLoss() {}

  MultiRationalLoss(const RationalLoss& rl) : RationalLoss() {
    push_back(rl);
  }

  virtual void push_back(const RationalLoss& current) {
    loss_   = accumulateLoss(current);
    count_ =  accumulateCount(current);
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
 * @brief Simple sum of losses.
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

  virtual Expr accumulateCount(const RationalLoss& current) override {
    if(count_)
      return count_ + current.count();
    else
      return current.count();
  }

public:
  SumMultiRationalLoss() : MultiRationalLoss() {}
  SumMultiRationalLoss(const RationalLoss& rl) : MultiRationalLoss(rl) {}
};

/**
 * @brief Scaled sum of losses.
 * This can weigh losses equally by choosing the first loss_0 as a reference
 * and scaling all remaining losses loss_i by count_0 / count_i. Labels are
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
      return loss_ + first.count() * (current.loss() / current.count()); // scale up/down to match scale of first loss
    } else {
      return current.loss(); // first reference loss, keeps to scale with this one
    }
  }

  virtual Expr accumulateCount(const RationalLoss& current) override {
    if(count_) {
      return count_; // Keep first label count // or: count_ + first.count() / current.count();
    } else {
      return current.count(); // This is the first loss
    }
  }

public:
  ScaledMultiRationalLoss() : MultiRationalLoss() {}
  ScaledMultiRationalLoss(const RationalLoss& rl) : MultiRationalLoss(rl) {}
};

/**
 * @brief Sum of mean losses.
 * Not really a rational loss as labels are factored into loss. Contribution of
 * losses is equal, same as for ScaledMultiRationalLoss, just divided by different
 * number of labels. See:
 *
 * L = (1/N sum_i^N L_i + 1/M sum_j^M L_j) = (sum_i^N L_i + N/M sum_j^M L_j) / N
 *
 * We set labels to 1. During reporting, we would see the same numbers, but gradients
 * are scaled differently which may result in different learning curves.
 */
class MeanMultiRationalLoss : public MultiRationalLoss {
private:
  virtual Expr accumulateLoss(const RationalLoss& current) override {
    if(loss_)
      return loss_ + current.loss() / current.count();
    else
      return current.loss() / current.count();
  }

  virtual Expr accumulateCount(const RationalLoss& current) override {
    if(count_)
      return count_; // keep the existing '1'
    else
      return current.count()->graph()->ones({1}); // just '1' as labels are factored into loss_
  }

public:
  MeanMultiRationalLoss() : MultiRationalLoss() {}
  MeanMultiRationalLoss(const RationalLoss& rl) : MultiRationalLoss(rl) {}
};

/**
 * @brief Factory for multi-objective rational loss functions
 */
Ptr<MultiRationalLoss> newMultiLoss(Ptr<Options> options);

//***********************************************************************************//
// This needs to be refactored. Currently easiest route for backwards compat, but
// still feels somewhat hacky.

/**
 * @brief Computes loss per given groundtruth label and then reduces to RationalLoss
 */
class LabelwiseLoss {
protected:
  std::vector<int> axes_;

  virtual Expr compute(Expr logits, Expr labelIndices,
                       Expr mask = nullptr, Expr labelWeights = nullptr) = 0;

  // label counts are available, reduce together with loss to obtain counts
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

  // label counts are not available, assume every element of tensor corresponds to label count 1
  RationalLoss reduce(Expr loss) {
    ABORT_IF(!loss, "Loss has not been computed");

    Expr  lossSum    = loss;
    for(int i = 0; i < axes_.size(); ++i)
      lossSum = sum(lossSum, axes_[i]);

    // reduction factor tells how over how many labels we reduced in total.
    float reducedLabels = (float)loss->shape().elements() / (float)lossSum->shape().elements();
    return RationalLoss(lossSum, reducedLabels);
  }

public:
  LabelwiseLoss(const std::vector<int>& axes)
  : axes_(axes) { }

  virtual RationalLoss apply(Expr logits, Expr labelIndices,
                             Expr mask = nullptr, Expr labelWeights = nullptr) {
    Expr loss = compute(logits, labelIndices, mask, labelWeights);

    if(mask)
      return reduce(loss, mask); // mask can be used as element-wise label count with broadcasting
    else
      return reduce(loss); // we have no mask, assume all items are labels
  }
};

/**
 * @brief Cross entropy loss across last axis, summed up over batch and time dimensions
 */
class CrossEntropyLoss : public LabelwiseLoss {
public:
  CrossEntropyLoss(float labelSmoothing)
  : LabelwiseLoss(/*axes=*/{-2, -3}), // cross-entropy already reduces over axis -1
    labelSmoothing_(labelSmoothing) {}

  CrossEntropyLoss(const std::vector<int>& axes, float labelSmoothing)
  : LabelwiseLoss(axes), // cross-entropy already reduces over axis -1
    labelSmoothing_(labelSmoothing) {}

protected:
  float labelSmoothing_; // interpolation factor for label smoothing, see below

  virtual Expr compute(Expr logits, Expr labelIndices,
                       Expr mask = nullptr, Expr labelWeights = nullptr) override {
    logits = atleast_3d(logits); // we always assuma a time and batch dimension exists.
    // for bert training or classification the time dimension is lot.
    // Here safeguard against 2d classifier output, adds 1 on the left, non-op.

    Expr ce = cross_entropy(logits, labelIndices);

    if(labelSmoothing_ > 0) {
      // @TODO: add this to CE kernels instead

      // Label smoothing (see https://arxiv.org/pdf/1512.00567.pdf, section 7)
      // We compute smoothed H(q',p) = (1 - eps) * H(q,p) + eps * H(u,p) where H(q,p) is the normal cross-entropy
      // and H(u,p) penalizes deviation of p from u, u being uniform distribution over vocab V => u_v = 1/|V|.
      // H(u,p) = - \sum_{v \in V} u_v * \log p_v = - 1/|V| \sum_{v \in V} \log \softmax_v => -mean(logsoftmax(logits))
      // ceq = -H(u,p) - avoid one kernel call by negating in the interpolation below
      Expr ceq = mean(logsoftmax(logits), /*axis=*/ -1);

      // H(q',p) = (1 - eps) * H(q,p) - eps * -H(u,p)
      ce = (1 - labelSmoothing_) * ce - labelSmoothing_ * ceq;
    }

    if(mask)
      ce = ce * mask;

    if(labelWeights)
      ce = ce * labelWeights;

    return ce;
  }
};

/**
 * @brief Cross entropy in rescorer used for computing sentences-level log probabilities
 */
class RescorerLoss : public CrossEntropyLoss {
public:
  // sentence-wise CE, hence reduce only over time axis. CE reduces over last axis (-1)
  RescorerLoss() : CrossEntropyLoss(/*axes=*/{-3}, /*smoothing=*/0.f) {}

  virtual RationalLoss apply(Expr logits, Expr labelIndices,
                             Expr mask = nullptr, Expr labelWeights = nullptr) override {
    auto ce = CrossEntropyLoss::apply(logits, labelIndices, mask, labelWeights);
    return RationalLoss(ce.loss(), ce.count());
  }
};

/**
 * @brief Factory for label-wise loss functions
 */
Ptr<LabelwiseLoss> newLoss(Ptr<Options> options, bool inference);

}  // namespace marian
