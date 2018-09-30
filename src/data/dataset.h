#pragma once

#include "common/definitions.h"
#include "data/batch.h"
#include "data/rng_engine.h"
#include "data/vocab.h"
#include "training/training_state.h"

namespace marian {
namespace data {

template <class Sample, class Iterator, class Batch>
class DatasetBase {
protected:
  std::vector<std::string> paths_;

public:
  typedef Batch batch_type;
  typedef Ptr<Batch> batch_ptr;
  typedef Iterator iterator;
  typedef Sample sample;

  DatasetBase() {}
  DatasetBase(std::vector<std::string> paths) : paths_(paths) {}

  virtual Iterator begin() = 0;
  virtual Iterator end() = 0;
  virtual void shuffle() = 0;

  virtual Sample next() = 0;

  virtual batch_ptr toBatch(const std::vector<sample>&) = 0;

  virtual void reset() {}
  virtual void prepare() {}
  virtual void restore(Ptr<TrainingState>) {}
};


}  // namespace data
}  // namespace marian
