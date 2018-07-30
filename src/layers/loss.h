#pragma once

#include "marian.h"

namespace marian {
class LossBase {
protected:
  float smoothing_;

public:
  explicit LossBase(float smoothing = 0) : smoothing_(smoothing){};

  Expr getCrossEntropy(Expr logits, Expr indices, Expr mask, Expr weights);
  virtual Expr getCost(Expr logits,
                       Expr indices,
                       Expr mask,
                       Expr weights = nullptr)
      = 0;
};

/*
 * @brief The cross entropy loss function
 *
 * A sum over words and average over sentences
 */
class CrossEntropyMeanLoss : public LossBase {
public:
  explicit CrossEntropyMeanLoss(float smoothing = 0) : LossBase(smoothing){};
  Expr getCost(Expr logits, Expr indices, Expr mask, Expr weights);
};

/*
 * @brief The cross entropy loss function as an average over target tokens
 */
class CrossEntropyMeanWordsLoss : public LossBase {
public:
  explicit CrossEntropyMeanWordsLoss(float smoothing = 0)
      : LossBase(smoothing){};
  Expr getCost(Expr logits, Expr indices, Expr mask, Expr weights);
};

/*
 * @brief The cross entropy loss function as a sum over target tokens
 */
class CrossEntropySumLoss : public LossBase {
public:
  explicit CrossEntropySumLoss(float smoothing = 0) : LossBase(smoothing){};
  Expr getCost(Expr logits, Expr indices, Expr mask, Expr weights);
};

/*
 * @brief The perplexity loss function
 */
class PerplexityLoss : public LossBase {
public:
  explicit PerplexityLoss(float smoothing = 0) : LossBase(smoothing){};
  Expr getCost(Expr logits, Expr indices, Expr mask, Expr weights);
};

/*
 * @brief The cross entropy loss function that keeps sentence-level costs
 */
class CrossEntropyRescoreLoss : public LossBase {
public:
  explicit CrossEntropyRescoreLoss(float smoothing = 0) : LossBase(smoothing){};
  Expr getCost(Expr logits, Expr indices, Expr mask, Expr weights);
};

Ptr<LossBase> LossFactory(Ptr<Options> options, bool inference);
}
