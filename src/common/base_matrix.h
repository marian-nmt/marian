#pragma once
#include <string>
#include <vector>
#include <memory>
#include "common/types.h"

class Hypothesis;
typedef std::shared_ptr<Hypothesis> HypothesisPtr;
typedef std::vector<HypothesisPtr> Beam;

class Scorer;
typedef std::shared_ptr<Scorer> ScorerPtr;

class History;

///////////////////////////////////////////////////////////////////
class BaseMatrix;
typedef std::vector<BaseMatrix*> BaseMatrices;

class BaseMatrix {
  public:
    virtual ~BaseMatrix() {}

    virtual size_t Rows() const = 0;
    virtual size_t Cols() const = 0;
    virtual void Resize(size_t rows, size_t cols) = 0;

    virtual void BestHyps(Beam& bestHyps,
        const Beam& prevHyps,
        BaseMatrices& ProbsEnsemble,
        const size_t beamSize,
        History& history,
        const std::vector<ScorerPtr> &scorers,
        const Words &filterIndices,
        bool returnAlignment=false) const = 0;

    virtual std::string Debug() const = 0;
};
