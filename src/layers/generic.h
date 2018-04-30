#pragma once

#include "marian.h"

#include "layers/factory.h"
#include "data/shortlist.h"

namespace marian {
namespace mlp {
enum struct act : int { linear, tanh, logit, ReLU, LeakyReLU, PReLU, swish };
}
}

YAML_REGISTER_TYPE(marian::mlp::act, int)

namespace marian {
namespace mlp {

class Layer {
protected:
  Ptr<ExpressionGraph> graph_;
  Ptr<Options> options_;

public:
  Layer(Ptr<ExpressionGraph> graph, Ptr<Options> options)
      : graph_(graph), options_(options) {}

  template <typename T>
  T opt(const std::string key) {
    return options_->get<T>(key);
  }

  template <typename T>
  T opt(const std::string key, T defaultValue) {
    return options_->get<T>(key, defaultValue);
  }

  virtual Expr apply(const std::vector<Expr>&) = 0;
  virtual Expr apply(Expr) = 0;
};

class Dense : public Layer {
public:
  Dense(Ptr<ExpressionGraph> graph, Ptr<Options> options)
      : Layer(graph, options) {}

  Expr apply(const std::vector<Expr>& inputs) {
    ABORT_IF(inputs.empty(), "No inputs");

    auto name = opt<std::string>("prefix");
    auto dim = opt<int>("dim");

    auto layerNorm = opt<bool>("layer-normalization", false);
    auto nematusNorm = opt<bool>("nematus-normalization", false);
    auto activation = opt<act>("activation", act::linear);

    auto g = graph_;

    std::vector<Expr> outputs;
    size_t i = 0;

    std::string num;
    for(auto&& in : inputs) {
      if(inputs.size() > 1)
        num = std::to_string(i);

      Expr W = g->param(name + "_W" + num,
                        {in->shape()[-1], dim},
                        inits::glorot_uniform);
      Expr b = g->param(name + "_b" + num,
                        {1, dim},
                        inits::zeros);

      if(layerNorm) {
        if(nematusNorm) {
          auto ln_s = g->param(name + "_ln_s" + num,
                               {1, dim},
                               inits::from_value(1.f));
          auto ln_b = g->param(name + "_ln_b" + num,
                               {1, dim},
                               inits::zeros);

          outputs.push_back(layer_norm(affine(in, W, b), ln_s, ln_b, NEMATUS_LN_EPS));
        } else {
          auto gamma = g->param(name + "_gamma" + num,
                                {1, dim},
                                inits::from_value(1.0));

          outputs.push_back(layer_norm(dot(in, W), gamma, b));
        }

      } else {
        outputs.push_back(affine(in, W, b));
      }
      i++;
    }

    switch(activation) {
      case act::linear: return plus(outputs);
      case act::tanh: return tanh(outputs);
      case act::logit: return logit(outputs);
      case act::ReLU: return relu(outputs);
      case act::LeakyReLU: return leakyrelu(outputs);
      case act::PReLU: return prelu(outputs);
      case act::swish: return swish(outputs);
      default: return plus(outputs);
    }
  };

  Expr apply(Expr input) {
    return apply(std::vector<Expr>({input}));
  }
};

class Output : public Layer {
private:
  std::map<std::string, Expr> tiedParams_;
  Ptr<data::Shortlist> shortlist_;

  Expr W_;
  Expr b_;
  bool transposeW_{false};

public:
  Output(Ptr<ExpressionGraph> graph, Ptr<Options> options)
      : Layer(graph, options) {}

  void tie_transposed(const std::string& param, const std::string& tied) {
    tiedParams_[param] = graph_->get(tied);
  }

  void set_shortlist(Ptr<data::Shortlist> shortlist) {
    shortlist_ = shortlist;
  }

  Expr apply(Expr input) {
    if(!W_) {
      auto name = options_->get<std::string>("prefix");
      auto dim = options_->get<int>("dim");
      std::string nameW = "W";

      if(tiedParams_.count(nameW)) {
        transposeW_ = true;
        W_ = tiedParams_[nameW];
        if(shortlist_)
          W_ = rows(W_, shortlist_->indices());
      } else {
        W_ = graph_->param(name + "_" + nameW,
                          {input->shape()[-1], dim},
                          inits::glorot_uniform);
        if(shortlist_)
          W_ = cols(W_, shortlist_->indices());
      }

      b_ = graph_->param(name + "_b",
                         {1, dim},
                         inits::zeros);
      if(shortlist_)
        b_ = cols(b_, shortlist_->indices());
    }

    return affine(input, W_, b_, false, transposeW_);
  }

  virtual Expr apply(const std::vector<Expr>& inputs) {
    ABORT("Not implemented");
  };

};


}  // namespace mlp

struct EmbeddingFactory : public Factory {
  EmbeddingFactory(Ptr<ExpressionGraph> graph) : Factory(graph) {}

  Expr construct() {
    std::string name = opt<std::string>("prefix");
    int dimVoc = opt<int>("dimVocab");
    int dimEmb = opt<int>("dimEmb");

    bool fixed = opt<bool>("fixed", false);

    NodeInitializer initFunc = inits::glorot_uniform;
    if(options_->has("embFile")) {
      std::string file = opt<std::string>("embFile");
      if(!file.empty()) {
        bool norm = opt<bool>("normalization", false);
        initFunc = inits::from_word2vec(file, dimVoc, dimEmb, norm);
      }
    }

    return graph_->param(name, {dimVoc, dimEmb}, initFunc, fixed);
  }
};

typedef Accumulator<EmbeddingFactory> embedding;

static inline Expr Cost(Expr logits,
                        Expr indices,
                        Expr mask,
                        std::string costType = "cross-entropy",
                        float smoothing = 0,
                        Expr weights = nullptr) {
  using namespace keywords;

  auto ce = cross_entropy(logits, indices);

  if(weights)
    ce = weights * ce;

  if(smoothing > 0) {
    // @TODO: add this to CE kernels instead
    auto ceq = mean(logsoftmax(logits), axis = -1);
    ce = (1 - smoothing) * ce - smoothing * ceq;
  }

  if(mask)
    ce = ce * mask;

  auto costSum = sum(ce, axis = -3);

  Expr cost;
  // axes:
  //  - time axis (words): -3
  //  - batch axis (sentences): -2
  if(costType == "ce-mean"
     || costType
            == "cross-entropy") {  // sum over words; average over sentences
    cost = mean(costSum, axis = -2);
  } else if(costType == "ce-mean-words") {  // average over target tokens
    cost = sum(costSum, axis = -2) / sum(sum(mask, axis = -3), axis = -2);
  } else if(costType == "ce-sum") {  // sum over target tokens
    cost = sum(costSum, axis = -2);
  } else if(costType == "perplexity") {  // ==exp('ce-mean-words')
    cost = exp(sum(costSum, axis = -2) / sum(sum(mask, axis = -3), axis = -2));
  } else if(costType == "ce-rescore") {  // sum over words, keep batch axis
    cost = -costSum;
  } else {  // same as ce-mean
    cost = mean(costSum, axis = -2);
  }

  return cost;
}
}
