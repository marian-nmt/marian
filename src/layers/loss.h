#pragma once

#include "data/types.h"
#include "graph/expression_operators.h"
#include "layers/logits.h"  // for Logits (Frank's factor hack)

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
 *
 * @TODO: This seems also used to hold a pair of (logits, mask)
 */
class RationalLoss {
protected:
  Expr loss_;   // numerator
  Expr count_;  // denominator

  RationalLoss() = default;  // protected

public:
  RationalLoss(Expr loss, Expr count) : loss_(loss), count_(count) {}

  RationalLoss(Expr loss, float count)
      : loss_(loss), count_(constant_like(loss, inits::fromValue(count))) {}

  RationalLoss(const RationalLoss& other) : loss_(other.loss_), count_(other.count_) {}

  virtual ~RationalLoss() = default;

  Expr loss() const { return loss_; }

  // @TODO: remove this function, as it does not add too much value over loss()->get(...)
  template <typename T>
  void loss(std::vector<T>& losses) const {
    ABORT_IF(!loss_, "Loss has not been defined");
    loss_->val()->get(losses);
  }

  template <typename T>
  T loss() const {  // this will fail if loss is not a single value
    ABORT_IF(!loss_, "Loss has not been defined");
    return loss_->val()->scalar<T>();
  }

  Expr count() const { return count_; }

  // @TODO: remove this function, as it does not add too much value over count()->get(...)
  template <typename T>
  void count(std::vector<T>& labels) const {
    ABORT_IF(!count_, "Labels have not been defined");
    count_->val()->get(labels);
  }

  template <typename T>
  T count() const {  // this will fail if loss is not a single value
    ABORT_IF(!count_, "Labels have not been defined");
    return count_->val()->scalar<T>();
  }

  // @TODO: add a function for returning maybe ratio?

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
  float loss;   // numerator
  float count;  // denominator

  StaticLoss() : loss(0.f), count(0.f) {}

  StaticLoss(const RationalLoss& dynamic)
      : loss(dynamic.loss<float>()), count(dynamic.count<float>()) {}

  StaticLoss operator+(const StaticLoss& other) const {
    StaticLoss res(*this);
    res += other;
    return res;
  }

  StaticLoss& operator+=(const StaticLoss& other) {
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

  MultiRationalLoss(const RationalLoss& rl) : RationalLoss() { push_back(rl); }

  virtual void push_back(const RationalLoss& current) {
    loss_ = accumulateLoss(current);
    count_ = accumulateCount(current);
    partialLosses_.push_back(current);
  }

  const RationalLoss& operator[](size_t i) { return partialLosses_[i]; }

  auto begin() -> decltype(partialLosses_.begin()) const { return partialLosses_.begin(); }

  auto end() -> decltype(partialLosses_.end()) const { return partialLosses_.end(); }

  size_t size() const { return partialLosses_.size(); }
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
 * We set labels to N. When reporting L/N this is equivalent to sum of means.
 * Compare to sum of means below where N is factored into the loss, but labels
 * are set to 1.
 */
class ScaledMultiRationalLoss : public MultiRationalLoss {
private:
  virtual Expr accumulateLoss(const RationalLoss& current) override {
    if(loss_) {
      const auto& first = partialLosses_.front();
      return loss_
             + current.loss() * first.count()
                   / current.count();  // scale up/down to match scale of first loss
    } else {
      return current.loss();  // first reference loss, keeps to scale with this one
    }
  }

  virtual Expr accumulateCount(const RationalLoss& current) override {
    if(count_) {
      return count_;  // Keep first label count // or: count_ + first.count() / current.count();
    } else {
      return current.count();  // This is the first loss
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
      return count_;  // keep the existing '1'
    else
      return current.count()->graph()->ones(
          {1}, current.loss()->value_type());  // just '1' as labels are factored into loss_
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

  virtual Expr compute(Logits logits,
                       const Words& labels,
                       Expr mask = nullptr,
                       Expr labelWeights = nullptr)
      = 0;

  // label counts are available, reduce together with loss to obtain counts
  RationalLoss reduce(Expr loss, Expr labels) {
    ABORT_IF(!loss, "Loss has not been computed");
    ABORT_IF(!labels, "Labels have not been computed");

    Expr lossSum = cast(loss, Type::float32);      // accumulate in float32
    Expr labelsSum = cast(labels, Type::float32);  // accumulate in float32
    for(int i = 0; i < axes_.size(); ++i) {
      lossSum = sum(lossSum, axes_[i]);
      labelsSum = sum(labelsSum, axes_[i]);
    }

    return RationalLoss(lossSum, labelsSum);
  }

  // label counts are not available, assume every element of tensor corresponds to label count 1
  RationalLoss reduce(Expr loss) {
    ABORT_IF(!loss, "Loss has not been computed");

    Expr lossSum = cast(loss, Type::float32);
    for(int i = 0; i < axes_.size(); ++i)
      lossSum = sum(lossSum, axes_[i]);

    // reduction factor tells how over how many labels we reduced in total.
    float reducedLabels = (float)loss->shape().elements() / (float)lossSum->shape().elements();
    return RationalLoss(lossSum, reducedLabels);
  }

public:
  LabelwiseLoss(const std::vector<int>& axes) : axes_(axes) {}

  virtual RationalLoss apply(Logits logits,
                             const Words& labels,
                             Expr mask = nullptr,
                             Expr labelWeights = nullptr) {
    Expr loss = compute(logits, labels, mask, labelWeights);

    if(mask)
      return reduce(loss, mask);  // mask can be used as element-wise label count with broadcasting
    else
      return reduce(loss);  // we have no mask, assume all items are labels
  }
};

/**
 * @brief Cross entropy loss across last axis, summed up over batch and time dimensions
 */
class CrossEntropyLoss : public LabelwiseLoss {
public:
  CrossEntropyLoss(float labelSmoothing, float factorWeight)
      : CrossEntropyLoss(/*axes=*/{-2, -3}, labelSmoothing, factorWeight) {
  }  // cross-entropy already reduces over axis -1

  CrossEntropyLoss(const std::vector<int>& axes, float labelSmoothing, float factorWeight)
      : LabelwiseLoss(axes),  // cross-entropy already reduces over axis -1
        labelSmoothing_(labelSmoothing),
        factorWeight_(factorWeight) {}

  virtual ~CrossEntropyLoss() {}

protected:
  float labelSmoothing_;  // interpolation factor for label smoothing, see below
  float factorWeight_;    // give extra weight to factors

  virtual Expr compute(Logits logits,
                       const Words& labels,
                       Expr mask = nullptr,
                       Expr labelWeights = nullptr) override {
    // logits may be factored; in that case, the getLoss() function computes one loss for each, and
    // sums them up
    int inFactor = false;
    auto ce = logits.applyLossFunction(labels, [&](Expr logits, Expr indices) {
      logits = atleast_3d(logits);  // we always assume a time and batch dimension exists.
      // for bert training or classification the time dimension is lost.
      // Here safeguard against 2d classifier output, adds 1 on the left, non-op.

      Expr ce = cross_entropy(logits, indices, inFactor ? 0.f : labelSmoothing_, Type::float32);
      if(inFactor && factorWeight_ != 1.0f) {
        LOG_ONCE(info, "scaling factor losses with weight {}", factorWeight_);
        ce = ce * factorWeight_;
      }
      inFactor = true;
      return ce;
    });

    if(mask)
      ce = ce * cast(mask, Type::float32);

    if(labelWeights) {
      // We currently do not know how to use target factors and word-level label weights together
      bool wordlevel = labelWeights->shape()[-3]
                       > 1;  // Time-dimension is not trivially 1, hence we have word-level weights.
      ABORT_IF(wordlevel && logits.getNumFactorGroups() > 1,
               "CE loss with word-level label weights is not implemented for factors");
      ce = ce * cast(labelWeights, Type::float32);
    }

    return ce;
  }
};

/**
 * @brief Unlikelihood loss across last axis, summed up over batch and time dimensions. This is an
 * implementation of sequence-level unlikelihood loss from https://arxiv.org/abs/1908.04319.
 * We rely on word-level label weights where 1 is correct and 0 is marking an error. If there are
 * not zeros for a sentence it going to be trained with normal CE loss if there is at least one 0 it
 * is going to flip over to use SUL for that sentence to penalize the selected word.
 *
 * SUL is implemented as:
 * -log(gather(1 - softmax(logits), -1, indices))
 *
 * Factors are currently not supported.
 */
class SequenceUnlikelihoodLoss : public CrossEntropyLoss {
public:
  SequenceUnlikelihoodLoss(float labelSmoothing, float factorWeight)
      : CrossEntropyLoss(labelSmoothing, factorWeight) {
  }  // cross-entropy already reduces over axis -1

  SequenceUnlikelihoodLoss(const std::vector<int>& axes, float labelSmoothing, float factorWeight)
      : CrossEntropyLoss(axes, labelSmoothing, factorWeight) {}

protected:
  virtual Expr compute(Logits logits,
                       const Words& labels,
                       Expr mask = nullptr,
                       Expr labelWeights = nullptr) override {
    auto ce = CrossEntropyLoss::compute(
        logits, labels, mask, /*labelWeights=*/nullptr);  // don't pass label-weights to CE
    if(!labelWeights)
      return ce;  // for validation, @TODO: maybe put rather abort or LOG_ONCE(warn, ...)?

    // We currently do not know how to use target factors and word-level label weights together
    ABORT_IF(logits.getNumFactorGroups() > 1, "Unlikelihood loss is not implemented for factors");

    ABORT_IF(!mask, "mask is required");  // @TODO: check this, it seems weights for padding are by
                                          // default 1, which would make this obsolete.
    // use label weights, where 1 is GOOD and 0 is BAD. After inversion here, now 1 marks BAD, mask
    // again to eliminate padding (might be obsolete)
    auto errorMask = (1.f - cast(labelWeights, Type::float32)) * cast(mask, Type::float32);

    auto ceUl = logits.applyLossFunction(labels, [&](Expr logits, Expr indices) {
      return cast(unlikelihood(logits, indices), Type::float32);
    });

    // compute if want to use CE or UL. If there are no errors train with CE, otherwise train _only
    // on_ the errors with UL. This is the "mixed" training schedule from
    // https://arxiv.org/abs/1908.04319. Providing labels with or without error scores we can easily
    // switch between CE and UL.
    auto onlyCe = eq(sum(errorMask, /*axis=*/-3),
                     0.f);    // [1, 1, dimBatch, 1] - equal 1 if no errors are present
    ceUl = errorMask * ceUl;  // don't use for correct label or padding

    auto cost = onlyCe * ce + (1.f - onlyCe) * ceUl;  // ce or unlikelihood part are never
                                                      // simultanously used as cost per batch entry

    return cost;
  }
};

/**
 * @brief Cross entropy in rescorer used for computing sentences-level or word-level log
 * probabilities
 *
 * This class differs from CrossEntropy in the different 'axes' setting, and that label smoothing
 * is disabled.
 */
class RescorerLoss : public CrossEntropyLoss {
private:
  bool wordScores_{false};  // compute word-level log probabilities

public:
  // For sentence-wise CE reduce only over time axis.
  // For word-level CE do not reduce over any axis.
  RescorerLoss(bool wordScores)
      : CrossEntropyLoss(/*axes=*/wordScores ? std::vector<int>({}) : std::vector<int>({-3}),
                         /*smoothing=*/0.f,
                         /*factorWeight=*/1.0f),
        wordScores_(wordScores) {}

  virtual RationalLoss apply(Logits logits,
                             const Words& labels,
                             Expr mask = nullptr,
                             Expr labelWeights = nullptr) override {
    auto loss = CrossEntropyLoss::compute(logits, labels, mask, labelWeights);

    if(!wordScores_) {  // for sentence-level CE, reduce loss and labels as in cross-entropy
      return reduce(loss, mask);
    } else {  // for word-level CE, reduce labels only to get sentence lengths
      ABORT_IF(!loss, "Loss has not been computed");
      ABORT_IF(!mask, "Word-level CE from rescorer must have mask");

      Expr labelsSum = cast(mask, Type::float32);  // accumulate in float32
      labelsSum = sum(labelsSum, -3);              // reduce over time axis to get sentence lengths
      return RationalLoss(loss, labelsSum);
    }
  }
};

/**
 * @brief Factory for label-wise loss functions
 */
Ptr<LabelwiseLoss> newLoss(Ptr<Options> options, bool inference);

}  // namespace marian
