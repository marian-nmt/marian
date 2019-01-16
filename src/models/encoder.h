#pragma once

#include "marian.h"
#include "models/states.h"

namespace marian {

class EncoderBase {
protected:
  Ptr<Options> options_;
  std::string prefix_{"encoder"};
  bool inference_{false};
  size_t batchIndex_{0};
public: 
  EncoderBase(Ptr<Options> options)
      : options_(options),
        prefix_(options->get<std::string>("prefix", "encoder")),
        inference_(options->get<bool>("inference", false)),
        batchIndex_(options->get<size_t>("index", 0)) {}

  virtual ~EncoderBase() {}

  virtual Ptr<EncoderState> build(Ptr<ExpressionGraph>, Ptr<data::CorpusBatch>) = 0;

  template <typename T>
  T opt(const std::string& key) const {
    return options_->get<T>(key);
  }

  virtual void clear() = 0;
};

}  // namespace marian
