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

  virtual std::tuple<Expr, Expr> lookup(Ptr<ExpressionGraph> graph,
                                        Expr srcEmbeddings,
                                        Ptr<data::CorpusBatch> batch) {
    using namespace keywords;

    auto subBatch = (*batch)[batchIndex_];

    int dimBatch = subBatch->batchSize();
    int dimEmb = srcEmbeddings->shape()[-1];
    int dimWords = subBatch->batchWidth();

    auto chosenEmbeddings = rows(srcEmbeddings, subBatch->data());

    auto batchEmbeddings
        = reshape(chosenEmbeddings, {dimWords, dimBatch, dimEmb});
    auto batchMask = graph->constant({dimWords, dimBatch, 1},
                                     inits::from_vector(subBatch->mask()));

    return std::make_tuple(batchEmbeddings, batchMask);
  }

public:
  EncoderBase(Ptr<Options> options)
      : options_(options),
        prefix_(options->get<std::string>("prefix", "encoder")),
        inference_(options->get<bool>("inference", false)),
        batchIndex_(options->get<size_t>("index", 0)) {}

  virtual Ptr<EncoderState> build(Ptr<ExpressionGraph>, Ptr<data::CorpusBatch>)
      = 0;

  template <typename T>
  T opt(const std::string& key) {
    return options_->get<T>(key);
  }

  virtual void clear() = 0;
};

}
