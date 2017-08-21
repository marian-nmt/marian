#pragma once

#include "marian.h"

#include "models/s2s.h"

namespace marian {

class Nematus : public S2S {
public:
  template <class... Args>
  Nematus(Ptr<Config> options, Args... args) : S2S(options, args...) {
    UTIL_THROW_IF2(options_->get<std::string>("enc-type") != "bidirectional",
                   "--type nematus does not currently support other encoder "
                   "type than bidirectional, use --type s2s");
    UTIL_THROW_IF2(options_->get<int>("enc-depth") > 1,
                   "--type nematus does not currently support multiple encoder "
                   "layers, use --type s2s");
    UTIL_THROW_IF2(options_->get<bool>("skip"),
                   "--type nematus does not currently support skip connections, "
                   "use --type s2s");
    UTIL_THROW_IF2(options_->get<int>("dec-depth") > 1,
                   "--type nematus does not currently support multiple decoder "
                   "layers, use --type s2s");
    UTIL_THROW_IF2(options_->get<int>("dec-cell-high-depth") > 1,
                   "--type nematus does not currently support multiple decoder "
                   "high cells, use --type s2s");

    UTIL_THROW_IF2(options_->get<std::string>("enc-cell") != "gru-nematus",
                   "--type nematus does not currently support other rnn cells "
                   "than gru-nematus, use --type s2s");
    UTIL_THROW_IF2(options_->get<std::string>("dec-cell") != "gru-nematus",
                   "--type nematus does not currently support other rnn cells "
                   "than gru-nematus, use --type s2s");
  }

  void load(Ptr<ExpressionGraph> graph, const std::string& name) {
    using namespace keywords;

    LOG(info)->info("Loading model from {}", name);

    auto numpy = cnpy::npz_load(name);

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

    std::vector<std::string> suffixes = {"_U", "_Ux", "_b", "_bx"};
    for(int i = 1; i < options_->get<int>("enc-cell-depth"); ++i) {
      std::string num1 = std::to_string(i);
      std::string num2 = std::to_string(i + 1);
      for(auto suf : suffixes) {
        nameMap.insert({"encoder" + suf + "_drt_" + num1,
                        "encoder_bi_cell" + num2 + suf});
        nameMap.insert({"encoder_r" + suf + "_drt_" + num1,
                        "encoder_bi_r_cell" + num2 + suf});
      }
    }
    for(int i = 3; i <= options_->get<int>("dec-cell-base-depth"); ++i) {
      std::string num1 = std::to_string(i - 2);
      std::string num2 = std::to_string(i);
      for(auto suf : suffixes)
        nameMap.insert({"decoder" + suf + "_nl_drt_" + num1,
                        "decoder_cell" + num2 + suf});
    }

    // TODO: remove debugs
    //for(auto& t : nameMap)
      //std::cerr << t.first << "\t" << t.second << std::endl;

    graph->setReloaded(false);

    for(auto it : numpy) {
      auto name = it.first;

      // TODO: remove debugs
      LOG(info)->info("key= " + name);

      if(name == "decoder_c_tt")
        continue;
      if(name.substr(0, 8) == "special:")
        continue;

      Shape shape;
      if(numpy[name].shape.size() == 2) {
        shape.set(0, numpy[name].shape[0]);
        shape.set(1, numpy[name].shape[1]);
      } else if(numpy[name].shape.size() == 1) {
        shape.set(0, 1);
        shape.set(1, numpy[name].shape[0]);
      }

      // TODO: remove debugs
      std::stringstream ss;
      ss << shape;
      LOG(info)->info("  shape= " + ss.str());

      std::string pName = name;
      if(nameMap.count(name))
        pName = nameMap[name];

      // TODO: remove debugs
      LOG(info)->info("  pName= " + pName + ((pName == name) ? " EQUAL" : ""));

      graph->param(pName, shape, init = inits::from_numpy(numpy[name]));
    }

    graph->setReloaded(true);
  }

  void save(Ptr<ExpressionGraph> graph,
            const std::string& name,
            bool saveTranslatorConfig) {
    save(graph, name);

    if(saveTranslatorConfig) {
      YAML::Node amun;
      auto vocabs = options_->get<std::vector<std::string>>("vocabs");
      amun["source-vocab"] = vocabs[0];
      amun["target-vocab"] = vocabs[1];
      amun["devices"] = options_->get<std::vector<int>>("devices");
      amun["normalize"] = true;
      amun["beam-size"] = 12;
      amun["relative-paths"] = false;

      amun["scorers"]["F0"]["path"] = name;
      amun["scorers"]["F0"]["type"] = "NematusDeep";
      amun["weights"]["F0"] = 1.0f;

      OutputFileStream out(name + ".amun.yml");
      (std::ostream&)out << amun;
    }
  }

  void save(Ptr<ExpressionGraph> graph, const std::string& name) {
    LOG(info)->info("Saving model to {}", name);

    // @TODO: support training with nematus model
    UTIL_THROW2("Saving is not yet supported");

    //unsigned shape[2];
    //std::string mode = "w";

    //std::map<std::string, std::string> nameMap
        //= {{"decoder_cell1_U", "decoder_U"},
           //{"decoder_cell1_Ux", "decoder_Ux"},
           //{"decoder_cell1_W", "decoder_W"},
           //{"decoder_cell1_Wx", "decoder_Wx"},
           //{"decoder_cell1_b", "decoder_b"},
           //{"decoder_cell1_bx", "decoder_bx"},
           //{"decoder_cell2_U", "decoder_U_nl"},
           //{"decoder_cell2_Ux", "decoder_Ux_nl"},
           //{"decoder_cell2_W", "decoder_Wc"},
           //{"decoder_cell2_Wx", "decoder_Wcx"},
           //{"decoder_cell2_b", "decoder_b_nl"},
           //{"decoder_cell2_bx", "decoder_bx_nl"},
           //{"decoder_ff_logit_l1_W0", "ff_logit_prev_W"},
           //{"decoder_ff_logit_l1_W1", "ff_logit_lstm_W"},
           //{"decoder_ff_logit_l1_W2", "ff_logit_ctx_W"},
           //{"decoder_ff_logit_l1_b0", "ff_logit_prev_b"},
           //{"decoder_ff_logit_l1_b1", "ff_logit_lstm_b"},
           //{"decoder_ff_logit_l1_b2", "ff_logit_ctx_b"},
           //{"decoder_ff_logit_l1_gamma0", "ff_logit_l1_gamma0"},
           //{"decoder_ff_logit_l1_gamma1", "ff_logit_l1_gamma1"},
           //{"decoder_ff_logit_l1_gamma2", "ff_logit_l1_gamma2"},
           //{"decoder_ff_logit_l2_W", "ff_logit_W"},
           //{"decoder_ff_logit_l2_b", "ff_logit_b"},
           //{"decoder_ff_state_W", "ff_state_W"},
           //{"decoder_ff_state_b", "ff_state_b"},
           //{"decoder_ff_state_gamma", "ff_state_gamma"},
           //{"decoder_Wemb", "Wemb_dec"},
           //{"encoder_Wemb", "Wemb"},
           //{"encoder_bi_U", "encoder_U"},
           //{"encoder_bi_Ux", "encoder_Ux"},
           //{"encoder_bi_W", "encoder_W"},
           //{"encoder_bi_Wx", "encoder_Wx"},
           //{"encoder_bi_b", "encoder_b"},
           //{"encoder_bi_bx", "encoder_bx"},
           //{"encoder_bi_gamma1", "encoder_gamma1"},
           //{"encoder_bi_gamma2", "encoder_gamma2"},
           //{"encoder_bi_r_U", "encoder_r_U"},
           //{"encoder_bi_r_Ux", "encoder_r_Ux"},
           //{"encoder_bi_r_W", "encoder_r_W"},
           //{"encoder_bi_r_Wx", "encoder_r_Wx"},
           //{"encoder_bi_r_b", "encoder_r_b"},
           //{"encoder_bi_r_bx", "encoder_r_bx"},
           //{"encoder_bi_r_gamma1", "encoder_r_gamma1"},
           //{"encoder_bi_r_gamma2", "encoder_r_gamma2"}};

    //graph->getBackend()->setDevice(graph->getDevice());

    //for(auto p : graph->params()->getMap()) {
      //std::vector<float> v;
      //p.second->val() >> v;

      //unsigned dim;
      //if(p.second->shape()[0] == 1) {
        //shape[0] = p.second->shape()[1];
        //dim = 1;
      //} else {
        //shape[0] = p.second->shape()[0];
        //shape[1] = p.second->shape()[1];
        //dim = 2;
      //}

      //std::string pName = p.first;
      //if(nameMap.count(pName))
        //pName = nameMap[pName];

      //cnpy::npz_save(name, pName, v.data(), shape, dim, mode);
      //mode = "a";
    //}

    //float ctt = 0;
    //shape[0] = 1;
    //cnpy::npz_save(name, "decoder_c_tt", &ctt, shape, 1, mode);

    //options_->saveModelParameters(name);
  }
};
}
