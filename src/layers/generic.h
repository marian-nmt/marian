#pragma once

#include "marian.h"

#include "data/shortlist.h"
#include "layers/factory.h"

namespace marian {
namespace mlp {
/**
 * @brief Activation functions
 */
enum struct act : int { linear, tanh, sigmoid, ReLU, LeakyReLU, PReLU, swish };
}  // namespace mlp
}  // namespace marian

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

  Expr apply(const std::vector<Expr>& inputs) override {
    ABORT_IF(inputs.empty(), "No inputs");

    auto name = opt<std::string>("prefix");
    auto dim = opt<int>("dim");

    auto useLayerNorm = opt<bool>("layer-normalization", false);
    auto useNematusNorm = opt<bool>("nematus-normalization", false);
    auto activation = opt<act>("activation", act::linear);

    auto g = graph_;

    std::vector<Expr> outputs;
    size_t i = 0;

    std::string num;
    for(auto&& in : inputs) {
      if(inputs.size() > 1)
        num = std::to_string(i);

      Expr W = g->param(
          name + "_W" + num, {in->shape()[-1], dim}, inits::glorot_uniform);
      Expr b = g->param(name + "_b" + num, {1, dim}, inits::zeros);

      if(useLayerNorm) {
        if(useNematusNorm) {
          auto ln_s = g->param(
              name + "_ln_s" + num, {1, dim}, inits::from_value(1.f));
          auto ln_b = g->param(name + "_ln_b" + num, {1, dim}, inits::zeros);

          outputs.push_back(
              layerNorm(affine(in, W, b), ln_s, ln_b, NEMATUS_LN_EPS));
        } else {
          auto gamma = g->param(
              name + "_gamma" + num, {1, dim}, inits::from_value(1.0));

          outputs.push_back(layerNorm(dot(in, W), gamma, b));
        }

      } else {
        outputs.push_back(affine(in, W, b));
      }
      i++;
    }

    // clang-format off
    switch(activation) {
      case act::linear:    return plus(outputs);
      case act::tanh:      return tanh(outputs);
      case act::sigmoid:   return sigmoid(outputs);
      case act::ReLU:      return relu(outputs);
      case act::LeakyReLU: return leakyrelu(outputs);
      case act::PReLU:     return prelu(outputs);
      case act::swish:     return swish(outputs);
      default:             return plus(outputs);
    }
    // clang-format on
  };

  Expr apply(Expr input) override { return apply(std::vector<Expr>({input})); }
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

  void set_shortlist(Ptr<data::Shortlist> shortlist) { shortlist_ = shortlist; }

  Expr apply(Expr input) override {
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

      b_ = graph_->param(name + "_b", {1, dim}, inits::zeros);
      if(shortlist_)
        b_ = cols(b_, shortlist_->indices());
    }

    return affine(input, W_, b_, false, transposeW_);
  }

  virtual Expr apply(const std::vector<Expr>& /*inputs*/) override {
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
  if (options_->has("embFile")) {
    std::string file = opt<std::string>("embFile");
    if (!file.empty()) {
      bool norm = opt<bool>("normalization", false);
      initFunc = inits::from_word2vec(file, dimVoc, dimEmb, norm);
    }
  }
  
    return graph_->param(name, {dimVoc, dimEmb}, initFunc, fixed);
  }
};


struct ULREmbeddingFactory : public Factory {
ULREmbeddingFactory(Ptr<ExpressionGraph> graph) : Factory(graph) {}

  std::vector<Expr> construct() {
    std::string name = "url_embed"; //opt<std::string>("prefix");
    int dimKeys = opt<int>("dimTgtVoc");
    int dimQueries = opt<int>("dimSrcVoc");
    int dimEmb = opt<int>("dimEmb");
    int dimUlrEmb =  opt<int>("dimUlrEmb"); // ULR mono embed size
    bool fixed = opt<bool>("fixed", false);
    std::vector<Expr> ulrEmbeds;
    NodeInitializer initFunc = inits::glorot_uniform;
    std::string queryFile = opt<std::string>("ulrQueryFile");
    std::string keyFile = opt<std::string>("ulrKeysFile");
    bool trainTrans = opt<bool>("ulrTrainTransform", false);
    if (!queryFile.empty() && !keyFile.empty()) {
      initFunc = inits::from_word2vec(queryFile, dimQueries, dimUlrEmb, false);
      name = "ulr_query";
      fixed = true;
      auto query_embed = graph_->param(name, { dimQueries, dimUlrEmb }, initFunc, fixed);
      ulrEmbeds.push_back(query_embed);
      // keys embeds
      initFunc = inits::from_word2vec(keyFile, dimKeys, dimUlrEmb, false);
      name = "ulr_keys";
      fixed = true;
      auto key_embed = graph_->param(name, { dimKeys, dimUlrEmb }, initFunc, fixed);
      ulrEmbeds.push_back(key_embed);
      // actual  trainable embedding
      initFunc = inits::glorot_uniform;
      name = "ulr_embed";
      fixed = false;
      auto ulr_embed = graph_->param(name, {dimKeys , dimEmb }, initFunc, fixed);  // note the reverse dim
      ulrEmbeds.push_back(ulr_embed);
      // init  trainable src embedding
      name = "ulr_src_embed";
      auto ulr_src_embed = graph_->param(name, { dimQueries, dimEmb }, initFunc, fixed);
      ulrEmbeds.push_back(ulr_src_embed);
      // ulr transformation matrix
      //initFunc = inits::eye(1.f); // identity matrix  - is it ok to init wiht identity or shall we make this to the fixed case only
      if (trainTrans) {
        initFunc = inits::glorot_uniform;
        fixed = false;
      }
      else
      {
        initFunc = inits::eye(); // identity matrix
        fixed = true;
      }
      name = "ulr_transform";
      auto ulr_transform = graph_->param(name, { dimUlrEmb, dimUlrEmb }, initFunc, fixed);
      ulrEmbeds.push_back(ulr_transform);

      initFunc = inits::from_value(1.f);  // TBD: we should read sharable flags here - 1 means all sharable - 0 means no universal embeddings - should be zero for top freq only
      fixed = true;
      name = "ulr_shared";
      auto share_embed = graph_->param(name, { dimQueries, 1 }, initFunc, fixed);
      ulrEmbeds.push_back(share_embed);

    }

    return ulrEmbeds;
  }
};

typedef Accumulator<EmbeddingFactory> embedding;
typedef Accumulator<ULREmbeddingFactory> ulr_embedding;
}  // namespace marian
