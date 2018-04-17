#pragma once

#include <string>
#include <vector>
#include <memory>
#include "common/types.h"

namespace amunmt {

const unsigned SHAPE_SIZE = 4;

class BaseTensor {
  public:
	BaseTensor() {}
    virtual ~BaseTensor() {}

    virtual unsigned dim(unsigned i) const = 0;

    virtual unsigned size() const;

    bool empty() const {
      return size() == 0;
    }

    virtual void Resize(unsigned rows, unsigned cols, unsigned beam = 1, unsigned batches = 1) = 0;

    virtual std::string Debug(unsigned verbosity = 1) const;
};

}

