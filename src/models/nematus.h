#pragma once

#include "marian.h"

#include "models/s2s.h"

namespace marian {

class Nematus : public EncoderDecoder {
public:
  template <class... Args>
  Nematus(Ptr<Options> options)
      : EncoderDecoder(options), nameMap_(createNameMap()) {
    ABORT_IF(options_->get<std::string>("enc-type") != "bidirectional",
             "--type nematus does not currently support other encoder "
             "type than bidirectional, use --type s2s");

    ABORT_IF(options_->get<std::string>("enc-cell") != "gru-nematus",
             "--type nematus does not currently support other rnn cells "
             "than gru-nematus, use --type s2s");
    ABORT_IF(options_->get<std::string>("dec-cell") != "gru-nematus",
             "--type nematus does not currently support other rnn cells "
             "than gru-nematus, use --type s2s");

    ABORT_IF(options_->get<int>("dec-cell-high-depth") > 1,
             "--type nematus does not currently support "
             "--dec-cell-high-depth > 1, use --type s2s");
  }

  void load(Ptr<ExpressionGraph> graph,
            const std::string& name,
            bool markedReloaded = true) {
    using namespace keywords;

    LOG(info, "Loading model from {}", name);
    auto numpy = cnpy::npz_load(name);

    graph->setReloaded(false);

    for(auto it : numpy) {
      auto name = it.first;

      if(name == "decoder_c_tt")
        continue;
      if(name.substr(0, 8) == "special:")
        continue;

      Shape shape;
      if(numpy[name]->shape.size() == 2) {
        shape.resize(2);
        shape.set(0, numpy[name]->shape[0]);
        shape.set(1, numpy[name]->shape[1]);
      } else if(numpy[name]->shape.size() == 1) {
        shape.resize(2);
        shape.set(0, 1);
        shape.set(1, numpy[name]->shape[0]);
      }

      std::string pName = name;
      if(nameMap_.count(name))
        pName = nameMap_[name];

      graph->param(pName, shape, inits::from_numpy(numpy[name]));
    }

    graph->setReloaded(true);
  }

  void save(Ptr<ExpressionGraph> graph,
            const std::string& name,
            bool saveTranslatorConfig = false) {
    LOG(info, "Saving model to {}", name);

    unsigned shape[2];
    std::string mode = "w";

    if(nameMapRev_.empty())
      for(auto& kv : nameMap_)
        nameMapRev_.insert({kv.second, kv.first});

    for(auto p : graph->params()->getMap()) {
      std::vector<float> v;
      p.second->val()->get(v);

      unsigned dim;
      if(p.second->shape()[0] == 1) {
        shape[0] = p.second->shape()[1];
        dim = 1;
      } else {
        shape[0] = p.second->shape()[0];
        shape[1] = p.second->shape()[1];
        dim = 2;
      }

      std::string pName = p.first;
      if(nameMapRev_.count(pName))
        pName = nameMapRev_[pName];

      cnpy::npz_save(name, pName, v.data(), shape, dim, mode);
      mode = "a";
    }

    float ctt = 0;
    shape[0] = 1;
    cnpy::npz_save(name, "decoder_c_tt", &ctt, shape, 1, mode);

    saveModelParameters(name);

    if(saveTranslatorConfig) {
      createAmunConfig(name);
      createDecoderConfig(name);
    }
  }

private:
  std::map<std::string, std::string> nameMap_;
  std::map<std::string, std::string> nameMapRev_;

  std::map<std::string, std::string> createNameMap() {
    std::map<std::string, std::string> nameMap
        = {{"decoder_U", "decoder_cell1_U"},
           {"decoder_Ux", "decoder_cell1_Ux"},
           {"decoder_W", "decoder_cell1_W"},
           {"decoder_Wx", "decoder_cell1_Wx"},
           {"decoder_b", "decoder_cell1_b"},
           {"decoder_bx", "decoder_cell1_bx"},
           {"decoder_U_nl", "decoder_cell2_U"},
           {"decoder_Ux_nl", "decoder_cell2_Ux"},
           {"decoder_Wc", "decoder_cell2_W"},
           {"decoder_Wcx", "decoder_cell2_Wx"},
           {"decoder_b_nl", "decoder_cell2_b"},
           {"decoder_bx_nl", "decoder_cell2_bx"},
           {"ff_logit_prev_W", "decoder_ff_logit_l1_W0"},
           {"ff_logit_lstm_W", "decoder_ff_logit_l1_W1"},
           {"ff_logit_ctx_W", "decoder_ff_logit_l1_W2"},
           {"ff_logit_prev_b", "decoder_ff_logit_l1_b0"},
           {"ff_logit_lstm_b", "decoder_ff_logit_l1_b1"},
           {"ff_logit_ctx_b", "decoder_ff_logit_l1_b2"},
           {"ff_logit_W", "decoder_ff_logit_l2_W"},
           {"ff_logit_b", "decoder_ff_logit_l2_b"},
           {"ff_state_W", "decoder_ff_state_W"},
           {"ff_state_b", "decoder_ff_state_b"},
           {"Wemb_dec", "decoder_Wemb"},
           {"Wemb", "encoder_Wemb"},
           {"encoder_U", "encoder_bi_U"},
           {"encoder_Ux", "encoder_bi_Ux"},
           {"encoder_W", "encoder_bi_W"},
           {"encoder_Wx", "encoder_bi_Wx"},
           {"encoder_b", "encoder_bi_b"},
           {"encoder_bx", "encoder_bi_bx"},
           {"encoder_r_U", "encoder_bi_r_U"},
           {"encoder_r_Ux", "encoder_bi_r_Ux"},
           {"encoder_r_W", "encoder_bi_r_W"},
           {"encoder_r_Wx", "encoder_bi_r_Wx"},
           {"encoder_r_b", "encoder_bi_r_b"},
           {"encoder_r_bx", "encoder_bi_r_bx"},
           {"ff_state_ln_s", "decoder_ff_state_ln_s"},
           {"ff_state_ln_b", "decoder_ff_state_ln_b"},
           {"ff_logit_prev_ln_s", "decoder_ff_logit_l1_ln_s0"},
           {"ff_logit_lstm_ln_s", "decoder_ff_logit_l1_ln_s1"},
           {"ff_logit_ctx_ln_s", "decoder_ff_logit_l1_ln_s2"},
           {"ff_logit_prev_ln_b", "decoder_ff_logit_l1_ln_b0"},
           {"ff_logit_lstm_ln_b", "decoder_ff_logit_l1_ln_b1"},
           {"ff_logit_ctx_ln_b", "decoder_ff_logit_l1_ln_b2"}};

    // add mapping for deep encoder cells
    std::vector<std::string> suffixes = {"_U", "_Ux", "_b", "_bx"};
    for(int i = 1; i < options_->get<int>("enc-cell-depth"); ++i) {
      std::string num1 = std::to_string(i);
      std::string num2 = std::to_string(i + 1);
      for(auto suf : suffixes) {
        nameMap.insert(
            {"encoder" + suf + "_drt_" + num1, "encoder_bi_cell" + num2 + suf});
        nameMap.insert({"encoder_r" + suf + "_drt_" + num1,
                        "encoder_bi_r_cell" + num2 + suf});
      }
    }
    // add mapping for deep decoder cells
    for(int i = 3; i <= options_->get<int>("dec-cell-base-depth"); ++i) {
      std::string num1 = std::to_string(i - 2);
      std::string num2 = std::to_string(i);
      for(auto suf : suffixes)
        nameMap.insert(
            {"decoder" + suf + "_nl_drt_" + num1, "decoder_cell" + num2 + suf});
    }
    // add mapping for normalization layers
    std::map<std::string, std::string> nameMapCopy(nameMap);
    for(auto& kv : nameMapCopy) {
      std::string prefix = kv.first.substr(0, 7);

      if(prefix == "encoder" || prefix == "decoder") {
        nameMap.insert({kv.first + "_lns", kv.second + "_lns"});
        nameMap.insert({kv.first + "_lnb", kv.second + "_lnb"});
      }
    }

    return nameMap;
  }

  void createAmunConfig(const std::string& name) {
    Config::YamlNode amun;
    // Amun has only CPU decoder for deep Nematus models
    amun["cpu-threads"] = 16;
    amun["gpu-threads"] = 0;
    amun["maxi-batch"] = 1;
    amun["mini-batch"] = 1;

    auto vocabs = options_->get<std::vector<std::string>>("vocabs");
    amun["source-vocab"] = vocabs[0];
    amun["target-vocab"] = vocabs[1];
    amun["devices"] = options_->get<std::vector<size_t>>("devices");
    amun["normalize"] = true;
    amun["beam-size"] = 5;
    amun["relative-paths"] = false;

    amun["scorers"]["F0"]["path"] = name;
    amun["scorers"]["F0"]["type"] = "nematus2";
    amun["weights"]["F0"] = 1.0f;

    OutputFileStream out(name + ".amun.yml");
    (std::ostream&)out << amun;
  }
};
}
