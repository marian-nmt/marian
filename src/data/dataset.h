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
  // Data processing may differ in training/inference settings
  std::vector<std::string> paths_;

  Ptr<Config> options_;
  bool inference_{false};


public:
  typedef Batch batch_type;
  typedef Ptr<Batch> batch_ptr;
  typedef Iterator iterator;
  typedef Sample sample;

  // @TODO: get rid of Config in favor of Options!
  DatasetBase(std::vector<std::string> paths, Ptr<Config> options)
    : paths_(paths), options_(options), inference_(options->get<bool>("inference", false)) {}

  DatasetBase(Ptr<Config> options) : DatasetBase({}, options) {}


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
