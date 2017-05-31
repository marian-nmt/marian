#pragma once

#include <string>
#include <vector>
#include <memory>
#include "common/types.h"

namespace amunmt {

const size_t SHAPE_SIZE = 4;

class Hypothesis;
typedef std::shared_ptr<Hypothesis> HypothesisPtr;
typedef std::vector<HypothesisPtr> Beam;

class Scorer;
typedef std::shared_ptr<Scorer> ScorerPtr;

class BaseMatrix {
  public:
	BaseMatrix() {}
    virtual ~BaseMatrix() {}

    virtual size_t dim(size_t i) const = 0;

    virtual size_t size() const;

    bool empty() const {
      return size() == 0;
    }

    virtual void Resize(size_t rows, size_t cols, size_t beam = 1, size_t batches = 1) = 0;

    virtual std::string Debug(size_t verbosity = 1) const;
};

}

