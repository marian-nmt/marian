#pragma once
#include <vector>
#include <memory>
#include "common/types.h"

class Hypothesis;
typedef std::shared_ptr<Hypothesis> HypothesisPtr;
typedef std::vector<HypothesisPtr> Beam;

class Scorer;
typedef std::shared_ptr<Scorer> ScorerPtr;

class History;

namespace mblas {

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
    		mblas::BaseMatrices& ProbsEnsemble,
    		const size_t beamSize,
    		History& history,
    		const std::vector<ScorerPtr> &scorers,
    		const Words &filterIndices) const = 0;

};

}

