#pragma once

#include "marian.h"

#include "models/s2s.h"

namespace marian {

class Nematus : public EncoderDecoder {
public:
  Nematus(Ptr<ExpressionGraph> graph, Ptr<Options> options) : EncoderDecoder(graph, options), nameMap_(createNameMap()) {
    ABORT_IF(options_->get<std::string>("enc-type") != "bidirectional",
             "--type nematus does not support other encoder "
             "type than bidirectional, use --type s2s");

    ABORT_IF(options_->get<std::string>("enc-cell") != "gru-nematus",
             "--type nematus does not support other rnn cells "
             "than gru-nematus, use --type s2s");
    ABORT_IF(options_->get<std::string>("dec-cell") != "gru-nematus",
             "--type nematus does not support other rnn cells "
             "than gru-nematus, use --type s2s");

    ABORT_IF(options_->get<int>("dec-cell-high-depth") > 1,
             "--type nematus does not currently support "
             "--dec-cell-high-depth > 1, use --type s2s");
  }

  void load(Ptr<ExpressionGraph> graph,
            const std::vector<io::Item>& items,
            bool /*markReloaded*/ = true) override {
    auto ioItems = items;
    // map names and remove a dummy matrix 'decoder_c_tt' from items to avoid creating isolated node
    for(auto it = ioItems.begin(); it != ioItems.end();) {
      // for backwards compatibility, turn one-dimensional vector into two dimensional matrix with first dimension being 1 and second dimension of the original size
      // @TODO: consider dropping support for Nematus models
      if(it->shape.size() == 1) {
        int dim = it->shape[-1];
        it->shape.resize(2);
        it->shape.set(0, 1);
        it->shape.set(1, dim);
      }

      if(it->name == "decoder_c_tt") {
        it = ioItems.erase(it);
      } else if(it->name == "uidx") {
        it = ioItems.erase(it);
      } else if(it->name == "history_errs") {
        it = ioItems.erase(it);
      } else {
        auto pair = nameMap_.find(it->name);
        if(pair != nameMap_.end())
          it->name = pair->second;
        it++;
      }
    }
    // load items into the graph
    graph->load(ioItems);
  }

  void load(Ptr<ExpressionGraph> graph,
            const std::string& name,
            bool /*markReloaded*/ = true) override {
    LOG(info, "Loading model from {}", name);
    auto ioItems = io::loadItems(name);
    load(graph, ioItems);
  }

  void save(Ptr<ExpressionGraph> graph,
            const std::string& name,
            bool saveTranslatorConfig = false) override {
    LOG(info, "Saving model to {}", name);

    // prepare reversed map
    if(nameMapRev_.empty())
      for(const auto& kv : nameMap_)
        nameMapRev_.insert({kv.second, kv.first});

    // get parameters from the graph to items
    std::vector<io::Item> ioItems;
    graph->save(ioItems);
    // replace names to be compatible with Nematus
    for(auto& item : ioItems) {
      auto newItemName = nameMapRev_.find(item.name);
      if(newItemName != nameMapRev_.end())
        item.name = newItemName->second;
    }
    // add a dummy matrix 'decoder_c_tt' required for Amun and Nematus
    ioItems.emplace_back();
    ioItems.back().name = "decoder_c_tt";
    ioItems.back().shape = Shape({1, 0});
    ioItems.back().bytes.emplace_back((char)0);

    io::addMetaToItems(getModelParametersAsString(), "special:model.yml", ioItems);
    io::saveItems(name, ioItems);

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
        nameMap.insert({"encoder" + suf + "_drt_" + num1, "encoder_bi_cell" + num2 + suf});
        nameMap.insert({"encoder_r" + suf + "_drt_" + num1, "encoder_bi_r_cell" + num2 + suf});
      }
    }
    // add mapping for deep decoder cells
    for(int i = 3; i <= options_->get<int>("dec-cell-base-depth"); ++i) {
      std::string num1 = std::to_string(i - 2);
      std::string num2 = std::to_string(i);
      for(auto suf : suffixes)
        nameMap.insert({"decoder" + suf + "_nl_drt_" + num1, "decoder_cell" + num2 + suf});
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

    io::OutputFileStream out(name + ".amun.yml");
    out << amun;
  }
};
}  // namespace marian
