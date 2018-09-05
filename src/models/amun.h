#pragma once

#include "marian.h"

#include "models/s2s.h"

namespace marian {

class Amun : public EncoderDecoder {
public:
  Amun(Ptr<Options> options) : EncoderDecoder(options) {
    ABORT_IF(opt<int>("enc-depth") > 1,
             "--type amun does not currently support multiple encoder "
             "layers, use --type s2s");
    ABORT_IF(opt<int>("enc-cell-depth") > 1,
             "--type amun does not currently support stacked encoder "
             "cells, use --type s2s");
    ABORT_IF(opt<bool>("skip"),
             "--type amun does not currently support skip connections, "
             "use --type s2s");
    ABORT_IF(opt<int>("dec-depth") > 1,
             "--type amun does not currently support multiple decoder "
             "layers, use --type s2s");
    ABORT_IF(opt<int>("dec-cell-base-depth") != 2,
             "--type amun does not currently support multiple decoder "
             "base cells, use --type s2s");
    ABORT_IF(opt<int>("dec-cell-high-depth") > 1,
             "--type amun does not currently support multiple decoder "
             "high cells, use --type s2s");
    ABORT_IF(opt<std::string>("enc-cell") != "gru",
             "--type amun does not currently support other rnn cells than gru, "
             "use --type s2s");
    ABORT_IF(opt<std::string>("dec-cell") != "gru",
             "--type amun does not currently support other rnn cells than gru, "
             "use --type s2s");
  }

  void load(Ptr<ExpressionGraph> graph,
            const std::string& name,
            bool /*markedReloaded*/ = true) override {
    std::map<std::string, std::string> nameMap
        = {{"decoder_U", "decoder_cell1_U"},
           {"decoder_Ux", "decoder_cell1_Ux"},
           {"decoder_W", "decoder_cell1_W"},
           {"decoder_Wx", "decoder_cell1_Wx"},
           {"decoder_b", "decoder_cell1_b"},
           {"decoder_bx", "decoder_cell1_bx"},
           {"decoder_cell1_gamma1", "decoder_cell1_gamma1"},
           {"decoder_cell1_gamma2", "decoder_cell1_gamma2"},
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
           {"ff_logit_l1_gamma0", "decoder_ff_logit_l1_gamma0"},
           {"ff_logit_l1_gamma1", "decoder_ff_logit_l1_gamma1"},
           {"ff_logit_l1_gamma2", "decoder_ff_logit_l1_gamma2"},
           {"ff_logit_W", "decoder_ff_logit_l2_W"},
           {"ff_logit_b", "decoder_ff_logit_l2_b"},
           {"ff_state_W", "decoder_ff_state_W"},
           {"ff_state_b", "decoder_ff_state_b"},
           {"ff_state_gamma", "decoder_ff_state_gamma"},
           {"Wemb_dec", "decoder_Wemb"},
           {"Wemb", "encoder_Wemb"},
           {"encoder_U", "encoder_bi_U"},
           {"encoder_Ux", "encoder_bi_Ux"},
           {"encoder_W", "encoder_bi_W"},
           {"encoder_Wx", "encoder_bi_Wx"},
           {"encoder_b", "encoder_bi_b"},
           {"encoder_bx", "encoder_bi_bx"},
           {"encoder_gamma1", "encoder_bi_gamma1"},
           {"encoder_gamma2", "encoder_bi_gamma2"},
           {"encoder_r_U", "encoder_bi_r_U"},
           {"encoder_r_Ux", "encoder_bi_r_Ux"},
           {"encoder_r_W", "encoder_bi_r_W"},
           {"encoder_r_Wx", "encoder_bi_r_Wx"},
           {"encoder_r_b", "encoder_bi_r_b"},
           {"encoder_r_bx", "encoder_bi_r_bx"},
           {"encoder_r_gamma1", "encoder_bi_r_gamma1"},
           {"encoder_r_gamma2", "encoder_bi_r_gamma2"}};

    if(opt<bool>("tied-embeddings-src") || opt<bool>("tied-embeddings-all"))
      nameMap["Wemb"] = "Wemb";

    graph->load(name, nameMap);
  }

  void save(Ptr<ExpressionGraph> graph,
            const std::string& name,
            bool saveTranslatorConfig = false) override {
    LOG(info, "Saving model to {}", name);

    std::map<std::string, std::string> nameMap
        = {{"decoder_cell1_U", "decoder_U"},
           {"decoder_cell1_Ux", "decoder_Ux"},
           {"decoder_cell1_W", "decoder_W"},
           {"decoder_cell1_Wx", "decoder_Wx"},
           {"decoder_cell1_b", "decoder_b"},
           {"decoder_cell1_bx", "decoder_bx"},
           {"decoder_cell2_U", "decoder_U_nl"},
           {"decoder_cell2_Ux", "decoder_Ux_nl"},
           {"decoder_cell2_W", "decoder_Wc"},
           {"decoder_cell2_Wx", "decoder_Wcx"},
           {"decoder_cell2_b", "decoder_b_nl"},
           {"decoder_cell2_bx", "decoder_bx_nl"},
           {"decoder_ff_logit_l1_W0", "ff_logit_prev_W"},
           {"decoder_ff_logit_l1_W1", "ff_logit_lstm_W"},
           {"decoder_ff_logit_l1_W2", "ff_logit_ctx_W"},
           {"decoder_ff_logit_l1_b0", "ff_logit_prev_b"},
           {"decoder_ff_logit_l1_b1", "ff_logit_lstm_b"},
           {"decoder_ff_logit_l1_b2", "ff_logit_ctx_b"},
           {"decoder_ff_logit_l1_gamma0", "ff_logit_l1_gamma0"},
           {"decoder_ff_logit_l1_gamma1", "ff_logit_l1_gamma1"},
           {"decoder_ff_logit_l1_gamma2", "ff_logit_l1_gamma2"},
           {"decoder_ff_logit_l2_W", "ff_logit_W"},
           {"decoder_ff_logit_l2_b", "ff_logit_b"},
           {"decoder_ff_state_W", "ff_state_W"},
           {"decoder_ff_state_b", "ff_state_b"},
           {"decoder_ff_state_gamma", "ff_state_gamma"},
           {"decoder_Wemb", "Wemb_dec"},
           {"encoder_Wemb", "Wemb"},
           {"encoder_bi_U", "encoder_U"},
           {"encoder_bi_Ux", "encoder_Ux"},
           {"encoder_bi_W", "encoder_W"},
           {"encoder_bi_Wx", "encoder_Wx"},
           {"encoder_bi_b", "encoder_b"},
           {"encoder_bi_bx", "encoder_bx"},
           {"encoder_bi_gamma1", "encoder_gamma1"},
           {"encoder_bi_gamma2", "encoder_gamma2"},
           {"encoder_bi_r_U", "encoder_r_U"},
           {"encoder_bi_r_Ux", "encoder_r_Ux"},
           {"encoder_bi_r_W", "encoder_r_W"},
           {"encoder_bi_r_Wx", "encoder_r_Wx"},
           {"encoder_bi_r_b", "encoder_r_b"},
           {"encoder_bi_r_bx", "encoder_r_bx"},
           {"encoder_bi_r_gamma1", "encoder_r_gamma1"},
           {"encoder_bi_r_gamma2", "encoder_r_gamma2"}};

    graph->save(name, getModelParametersAsString(), nameMap);

    if(saveTranslatorConfig) {
      createAmunConfig(name);
      createDecoderConfig(name);
    }
  }

private:
  void createAmunConfig(const std::string& name) {
    Config::YamlNode amun;
    auto vocabs = options_->get<std::vector<std::string>>("vocabs");
    amun["source-vocab"] = vocabs[0];
    amun["target-vocab"] = vocabs[1];
    amun["devices"] = options_->get<std::vector<size_t>>("devices");
    amun["normalize"] = opt<float>("normalize") > 0;
    amun["beam-size"] = opt<size_t>("beam-size");
    amun["relative-paths"] = false;

    amun["scorers"]["F0"]["path"] = name;
    amun["scorers"]["F0"]["type"] = "Nematus";
    amun["weights"]["F0"] = 1.0f;

    OutputFileStream out(name + ".amun.yml");
    (std::ostream&)out << amun;
  }
};
}  // namespace marian
