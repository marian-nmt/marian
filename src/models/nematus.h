#pragma once

#include "data/corpus.h"
#include "graph/expression_graph.h"
#include "layers/rnn.h"
#include "layers/param_initializers.h"
#include "layers/generic.h"
#include "3rd_party/cnpy/cnpy.h"

namespace marian {

class Nematus : public ExpressionGraph {
  public:
    Nematus()
    : dimSrcVoc_(40000), dimSrcEmb_(512), dimEncState_(1024),
      dimTrgVoc_(40000), dimTrgEmb_(512), dimDecState_(1024),
      dimBatch_(40) {}

    Nematus(const std::vector<int> dims)
    : dimSrcVoc_(dims[0]), dimSrcEmb_(dims[1]), dimEncState_(dims[2]),
      dimTrgVoc_(dims[3]), dimTrgEmb_(dims[4]), dimDecState_(dims[5]),
      dimBatch_(dims[6]) {}

  private:
    int dimSrcVoc_;
    int dimSrcEmb_;
    int dimEncState_;

    int dimTrgVoc_;
    int dimTrgEmb_;
    int dimDecState_;

    int dimBatch_;

    void setDims(const data::CorpusBatch& batch) {
      dimSrcVoc_ = this->get("Wemb") ? this->get("Wemb")->shape()[0] : dimSrcVoc_;
      dimSrcEmb_ = this->get("Wemb") ? this->get("Wemb")->shape()[1] : dimSrcEmb_;
      dimEncState_ = this->get("encoder_U") ? this->get("encoder_U")->shape()[0] : dimEncState_;

      dimTrgVoc_ = this->get("Wemb_dec") ? this->get("Wemb_dec")->shape()[0] : dimTrgVoc_;
      dimTrgEmb_ = this->get("Wemb_dec") ? this->get("Wemb_dec")->shape()[1] : dimTrgEmb_;
      dimDecState_ = this->get("decoder_U") ? this->get("decoder_U")->shape()[0] : dimDecState_;

      dimBatch_ = batch.size();
    }

  public:

    void load(const std::string& name) {
      using namespace keywords;

      std::cerr << "Loading model from " << name << std::endl;
      auto numpy = cnpy::npz_load(name);

      auto parameters = {
        // Source word embeddings
        "Wemb",

        // GRU in encoder
        "encoder_U", "encoder_W", "encoder_b",
        "encoder_Ux", "encoder_Wx", "encoder_bx",

        // GRU in encoder, reversed
        "encoder_r_U", "encoder_r_W", "encoder_r_b",
        "encoder_r_Ux", "encoder_r_Wx", "encoder_r_bx",

        // Transformation of decoder input state
        "ff_state_W", "ff_state_b",

        // Target word embeddings
        "Wemb_dec",

        // GRU layer 1 in decoder
        "decoder_U", "decoder_W", "decoder_b",
        "decoder_Ux", "decoder_Wx", "decoder_bx",

        // Attention
        "decoder_W_comb_att", "decoder_b_att",
        "decoder_Wc_att", "decoder_U_att",

        // GRU layer 2 in decoder
        "decoder_U_nl", "decoder_Wc", "decoder_b_nl",
        "decoder_Ux_nl", "decoder_Wcx", "decoder_bx_nl",

        // Read out
        "ff_logit_lstm_W", "ff_logit_lstm_b",
        "ff_logit_prev_W", "ff_logit_prev_b",
        "ff_logit_ctx_W", "ff_logit_ctx_b",
        "ff_logit_W", "ff_logit_b",
      };

      std::map<std::string, std::string> nameMap = {
        {"decoder_U", "decoder_gru1_U"},
        {"decoder_W", "decoder_gru1_W"},
        {"decoder_b", "decoder_gru1_b"},
        {"decoder_Ux", "decoder_gru1_Ux"},
        {"decoder_Wx", "decoder_gru1_Wx"},
        {"decoder_bx", "decoder_gru1_bx"},

        {"decoder_U_nl", "decoder_gru2_U"},
        {"decoder_Wc", "decoder_gru2_W"},
        {"decoder_b_nl", "decoder_gru2_b"},
        {"decoder_Ux_nl", "decoder_gru2_Ux"},
        {"decoder_Wcx", "decoder_gru2_Wx"},
        {"decoder_bx_nl", "decoder_gru2_bx"},

        {"ff_logit_prev_W", "ff_logit_l1_W0"},
        {"ff_logit_prev_b", "ff_logit_l1_b0"},
        {"ff_logit_lstm_W", "ff_logit_l1_W1"},
        {"ff_logit_lstm_b", "ff_logit_l1_b1"},
        {"ff_logit_ctx_W", "ff_logit_l1_W2"},
        {"ff_logit_ctx_b", "ff_logit_l1_b2"},

        {"ff_logit_W", "ff_logit_l2_W"},
        {"ff_logit_b", "ff_logit_l2_b"}
      };

      for(auto name : parameters) {
        Shape shape;
        if(numpy[name].shape.size() == 2) {
          shape.set(0, numpy[name].shape[0]);
          shape.set(1, numpy[name].shape[1]);
        }
        else if(numpy[name].shape.size() == 1) {
          shape.set(0, 1);
          shape.set(1, numpy[name].shape[0]);
        }

        std::string pName = name;
        if(nameMap.count(name))
          pName = nameMap[name];

        this->param(pName, shape,
                    init=inits::from_numpy(numpy[name]));
      }
    }

    void save(const std::string& name) {
      std::cerr << "Saving to " << name << std::endl;
      unsigned shape[2];
      std::string mode = "w";

      std::map<std::string, std::string> nameMap = {
        {"decoder_gru1_U", "decoder_U"},
        {"decoder_gru1_W", "decoder_W"},
        {"decoder_gru1_b", "decoder_b"},
        {"decoder_gru1_Ux", "decoder_Ux"},
        {"decoder_gru1_Wx", "decoder_Wx"},
        {"decoder_gru1_bx", "decoder_bx"},

        {"decoder_gru2_U", "decoder_U_nl"},
        {"decoder_gru2_W", "decoder_Wc"},
        {"decoder_gru2_b", "decoder_b_nl"},
        {"decoder_gru2_Ux", "decoder_Ux_nl"},
        {"decoder_gru2_Wx", "decoder_Wcx"},
        {"decoder_gru2_bx", "decoder_bx_nl"},

        {"ff_logit_l1_W0", "ff_logit_prev_W"},
        {"ff_logit_l1_b0", "ff_logit_prev_b"},
        {"ff_logit_l1_W1", "ff_logit_lstm_W"},
        {"ff_logit_l1_b1", "ff_logit_lstm_b"},
        {"ff_logit_l1_W2", "ff_logit_ctx_W"},
        {"ff_logit_l1_b2", "ff_logit_ctx_b"},

        {"ff_logit_l2_W", "ff_logit_W"},
        {"ff_logit_l2_b", "ff_logit_b"}
      };

      for(auto p : this->params().getMap()) {
        std::vector<float> v;
        p.second->val() >> v;

        unsigned dim;
        if(p.second->shape()[0] == 1) {
          shape[0] = p.second->shape()[1];
          dim = 1;
        }
        else {
          shape[0] = p.second->shape()[0];
          shape[1] = p.second->shape()[1];
          dim = 2;
        }

        std::string pName = p.first;
        if(nameMap.count(pName))
          pName = nameMap[pName];

        cnpy::npz_save(name, pName, v.data(), shape, dim, mode);
        mode = "a";
      }

      float ctt = 0;
      shape[0] = 1;
      cnpy::npz_save(name, "decoder_c_tt", &ctt, shape, 1, mode);
    }

    std::tuple<Expr, Expr>
    processSource(Expr Wemb, const data::SentBatch& srcSentenceBatch) {
      using namespace keywords;

      std::vector<size_t> indeces;
      std::vector<float> weightMask;

      std::vector<Expr> inputs;
      std::vector<std::pair<Expr, Expr>> inputsWithMask;

      for(auto& srcWordBatch : srcSentenceBatch) {
        for(auto i: srcWordBatch.first)
          indeces.push_back(i);
        for(auto m: srcWordBatch.second)
          weightMask.push_back(m);
      }

      int srcWords = (int)srcSentenceBatch.size();
      auto x = reshape(rows(Wemb, indeces), {dimBatch_, dimSrcEmb_, srcWords});
      auto xMask = this->constant(shape={dimBatch_, 1, srcWords},
                                  init=inits::from_vector(weightMask));

      return std::make_tuple(x, xMask);
    }

    std::tuple<Expr, Expr, Expr>
    processTarget(Expr Wemb_dec,
                  const data::SentBatch& trgSentenceBatch) {
      using namespace keywords;

      std::vector<float> weightMask;
      std::vector<float> picks;
      std::vector<size_t> indeces;
      for(int j = 0; j < trgSentenceBatch.size(); ++j) {
        auto& trgWordBatch = trgSentenceBatch[j];

        for(auto i : trgWordBatch.first) {
          picks.push_back((float)i);
          if(j < trgSentenceBatch.size() - 1)
            indeces.push_back(i);
        }

        for(auto m : trgWordBatch.second)
            weightMask.push_back(m);
      }

      int trgWords = (int)trgSentenceBatch.size();

      auto y = reshape(rows(Wemb_dec, indeces),
                       {dimBatch_, dimTrgEmb_, trgWords - 1});
      auto yMask = this->constant(shape={dimBatch_, 1, trgWords},
                                  init=inits::from_vector(weightMask));
      auto yPicks = this->constant(shape={(int)picks.size(), 1},
                                   init=inits::from_vector(picks));

      return std::make_tuple(y, yMask, yPicks);
    }

    Expr construct(const data::CorpusBatch& batch) {
      using namespace keywords;

      this->clear();
      setDims(batch);

      // Embeddings
      auto Wemb = Embedding("Wemb", dimSrcVoc_, dimSrcEmb_)
                    (this->shared_from_this());

      auto Wemb_dec = Embedding("Wemb_dec", dimTrgVoc_, dimTrgEmb_)
                        (this->shared_from_this());

      Expr x, xMask;
      std::tie(x, xMask) = processSource(Wemb, batch[0]);

      Expr y, yMask, yPicks;
      std::tie(y, yMask, yPicks) = processTarget(Wemb_dec, batch[1]);

      // Encoder
      auto xContext = BiRNN<GRU>("encoder", dimEncState_)
                        (x, mask=xMask);

      auto xMeanContext = weighted_average(xContext, xMask, axis=2);

      // Decoder
      auto yStart = Dense("ff_state",
                          dimDecState_,
                          activation=act::tanh)(xMeanContext);

      auto yEmpty = this->zeros(shape={dimBatch_, dimTrgEmb_});
      auto yShifted = concatenate({yEmpty, y}, axis=2);

      CGRU cgru({"decoder", xContext, dimDecState_, mask=xMask});
      auto yLstm = RNN<CGRU>("decoder", dimDecState_, cgru)
                     (yShifted, yStart);
      auto yCtx = cgru.getContexts();

      // 2-layer feedforward network for outputs and cost
      auto ff_logit_l1 = Dense("ff_logit_l1", dimTrgEmb_,
                               activation=act::tanh)
                           (yShifted, yLstm, yCtx);

      auto ff_logit_l2 = Dense("ff_logit_l2", dimTrgVoc_)
                           (ff_logit_l1);

      auto cost = CrossEntropyCost("cost")
                    (ff_logit_l2, yPicks, mask=yMask);

      return cost;
    }
};

}
