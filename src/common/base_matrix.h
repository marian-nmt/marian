#pragma once

#include <string>
#include <vector>
#include <memory>
#include "common/types.h"

namespace amunmt {

class Hypothesis;
typedef std::shared_ptr<Hypothesis> HypothesisPtr;
typedef std::vector<HypothesisPtr> Beam;

class Scorer;
typedef std::shared_ptr<Scorer> ScorerPtr;

class BaseMatrix {
  public:
	BaseMatrix() {}
	BaseMatrix(const BaseMatrix&) = delete;
    virtual ~BaseMatrix() {}

    virtual size_t Rows() const = 0;
    virtual size_t Cols() const = 0;
    virtual void Resize(size_t rows, size_t cols) = 0;

    virtual std::string Debug() const = 0;
};

}

