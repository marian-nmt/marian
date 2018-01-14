#pragma once

#include <string>
#include <vector>
#include <memory>
#include "common/types.h"

namespace amunmt {

const unsigned SHAPE_SIZE = 4;

class Hypothesis;
typedef std::shared_ptr<Hypothesis> HypothesisPtr;
typedef std::vector<HypothesisPtr> Beam;

class Scorer;
typedef std::shared_ptr<Scorer> ScorerPtr;

class BaseMatrix {
  public:
	BaseMatrix() {}
    virtual ~BaseMatrix() {}

    virtual unsigned dim(unsigned i) const = 0;

    virtual unsigned size() const;

    bool empty() const {
      return size() == 0;
    }

    virtual void Resize(unsigned rows, unsigned cols, unsigned beam = 1, unsigned batches = 1) = 0;

    virtual std::string Debug(unsigned verbosity = 1) const;
};

}

