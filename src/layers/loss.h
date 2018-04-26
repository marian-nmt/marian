#pragma once

#include "marian.h"
// #include "common/definitions.h"
// #include "common/options.h"
// #include "graph/expression_graph.h"
// #include "graph/expression_operators.h"
// #include "layers/factory.h"
// #include "layers/param_initializers.h"

namespace marian {
class LossBase {
protected:
  float smoothing_;

public:
  explicit LossBase(float smoothing = 0) : smoothing_(smoothing){};

  Expr getCrossEntropy(Expr logits, Expr indices, Expr mask, Expr weights);
  virtual Expr getCost(Expr logits, Expr indices, Expr mask, Expr weights = nullptr) = 0;
};

class CrossEntropyMeanLoss : public LossBase {
  /*
   * Sum over words; average over sentences
   */
public:
  explicit CrossEntropyMeanLoss(float smoothing = 0) : LossBase(smoothing){};
  Expr getCost(Expr logits, Expr indices, Expr mask, Expr weights);
};

class CrossEntropyMeanWordsLoss : public LossBase {
  /*
   * Average over target tokens.
   */
public:
  explicit CrossEntropyMeanWordsLoss(float smoothing = 0)
      : LossBase(smoothing){};
  Expr getCost(Expr logits, Expr indices, Expr mask, Expr weights);
};

class CrossEntropySumLoss : public LossBase {
public:
  /*
   * Sum over target tokens
   */
  explicit CrossEntropySumLoss(float smoothing = 0) : LossBase(smoothing){};
  Expr getCost(Expr logits, Expr indices, Expr mask, Expr weights);
};

class PerplexityLoss : public LossBase {
  /*
   * The same as exp(CrossEntropyMeanLoss)
   */
public:
  explicit PerplexityLoss(float smoothing = 0) : LossBase(smoothing){};
  Expr getCost(Expr logits, Expr indices, Expr mask, Expr weights);
};

class CrossEntropyRescoreLoss : public LossBase {
  /*
   * Sum over words, keep batch axis.
   */
public:
  explicit CrossEntropyRescoreLoss(float smoothing = 0) : LossBase(smoothing){};
  Expr getCost(Expr logits, Expr indices, Expr mask, Expr weights);
};

Ptr<LossBase> LossFactory(Ptr<Options> options, bool inference);
}
