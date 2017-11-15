#include "cpu/nematus/model.h"

namespace amunmt {
namespace CPU {
namespace Nematus {

Weights::Transition::Transition(const NpzConverter& model, TransitionType type, std::string prefix,
                                std::string infix)
  : depth_(findTransitionDepth(model, prefix, infix)), type_(type)
{
  for (int i = 1; i <= depth_; ++i) {
    U_.emplace_back(model[name(prefix, "U", infix, i)]);
    Ux_.emplace_back(model[name(prefix, "Ux", infix, i)]);
    B_.emplace_back(model(name(prefix, "b", infix, i), true));
    U_lns_.emplace_back(model[name(prefix, "U", infix, i, "_lns")]);
    U_lnb_.emplace_back(model[name(prefix, "U", infix, i, "_lnb")]);
    Ux_lns_.emplace_back(model[name(prefix, "Ux", infix, i, "_lns")]);
    Ux_lnb_.emplace_back(model[name(prefix, "Ux", infix, i, "_lnb")]);

    switch(type) {
      case TransitionType::Encoder:
        Bx1_.emplace_back(1, Ux_.back().dim(1));
        const_cast<mblas::Matrix&>(Bx1_.back()) = 0.0f;
        Bx2_.emplace_back(model(name(prefix, "bx", infix, i), true));
        break;
      case TransitionType::Decoder:
        Bx1_.emplace_back(model(name(prefix, "bx", infix, i), true));
        Bx2_.emplace_back(1, Ux_.back().dim(1));
        const_cast<mblas::Matrix&>(Bx2_.back()) = 0.0f;
        break;
    }
  }
}

int Weights::Transition::findTransitionDepth(const NpzConverter& model, std::string prefix, std::string infix) {
  int currentDepth = 0;
  while (true) {
    if (model.has(prefix + "b" + infix + "_drt_" + std::to_string(currentDepth + 1))) {
      ++currentDepth;
    } else {
      break;
    }
  }
  std::cerr << "Found transition depth: " << currentDepth << std::endl;
  return currentDepth;
}

int Weights::Transition::size() const {
  return depth_;
}

Weights::Transition::TransitionType Weights::Transition::type() const {
  return type_;
}


std::string Weights::Transition::name(const std::string& prefix, std::string name, std::string infix,
    int index, std::string suffix)
{
  return prefix + name + infix + "_drt_" + std::to_string(index) + suffix;
}

Weights::Embeddings::Embeddings(const NpzConverter& model, const std::string &key)
  : E_(model[key])
{}

Weights::Embeddings::Embeddings(const NpzConverter& model, const std::vector<std::pair<std::string, bool>> keys)
  : E_(model.getFirstOfMany(keys))
{}

Weights::GRU::GRU(const NpzConverter& model, std::string prefix, std::vector<std::string> keys)
  : W_(model[prefix + keys.at(0)]),
    B_(model(prefix + keys.at(1), true)),
    U_(model[prefix + keys.at(2)]),
    Wx_(model[prefix + keys.at(3)]),
    Bx1_(model(prefix + keys.at(4), true)),
    Bx2_(Bx1_.rows(), Bx1_.columns()),
    Bx3_(B_.rows(), B_.columns()),
    Ux_(model[prefix + keys.at(5)]),
    W_lns_(model[prefix + keys.at(6)]),
    W_lnb_(model[prefix + keys.at(7)]),
    Wx_lns_(model[prefix + keys.at(8)]),
    Wx_lnb_(model[prefix + keys.at(9)]),
    U_lns_(model[prefix + keys.at(10)]),
    U_lnb_(model[prefix + keys.at(11)]),
    Ux_lns_(model[prefix + keys.at(12)]),
    Ux_lnb_(model[prefix + keys.at(13)])
{
  const_cast<mblas::Matrix&>(Bx2_) = 0.0f;
  const_cast<mblas::Matrix&>(Bx3_) = 0.0f;
}

//////////////////////////////////////////////////////////////////////////////

Weights::DecInit::DecInit(const NpzConverter& model)
  : Wi_(model["ff_state_W"]),
    Bi_(model("ff_state_b", true)),
    lns_(model["ff_state_ln_s"]),
    lnb_(model["ff_state_ln_b"])
{}


Weights::DecGRU2::DecGRU2(const NpzConverter& model, std::string prefix, std::vector<std::string> keys)
  : W_(model[prefix + keys.at(0)]),  // Wc
    B_(1, W_.dim(1)),
    U_(model[prefix + keys.at(1)]),  // U_nl
    Bx3_(model(prefix + keys.at(2), true)),  // b_nl
    Wx_(model[prefix + keys.at(3)]),  // Wcx
    Bx1_(1, Wx_.dim(1)),
    Ux_(model[prefix + keys.at(4)]),  // Ux_nl
    Bx2_(model(prefix + keys.at(5), true)),  // bx_nl
    W_lns_(model[prefix + keys.at(6)]),  // Wc_lns
    W_lnb_(model[prefix + keys.at(7)]),  // Wc_nlb
    Wx_lns_(model[prefix + keys.at(8)]),  // Wcx_lns
    Wx_lnb_(model[prefix + keys.at(9)]),  // Wcx_lnb
    U_lns_(model[prefix + keys.at(10)]),  // U_nl_lns
    U_lnb_(model[prefix + keys.at(11)]),  // U_nl_lnb
    Ux_lns_(model[prefix + keys.at(12)]),  // Ux_nl_lns
    Ux_lnb_(model[prefix + keys.at(13)])  // Ux_nl_lnb

{
  const_cast<mblas::Matrix&>(B_) = 0.0f;
  const_cast<mblas::Matrix&>(Bx1_) = 0.0f;
}

Weights::DecAttention::DecAttention(const NpzConverter& model)
  : V_(model("decoder_U_att", true)),
    W_(model["decoder_W_comb_att"]),
    B_(model("decoder_b_att", true)),
    U_(model["decoder_Wc_att"]),
    C_(model["decoder_c_tt"]),
    Wc_att_lns_(model["decoder_Wc_att_lns"]),
    Wc_att_lnb_(model["decoder_Wc_att_lnb"]),
    W_comb_lns_(model["decoder_W_comb_att_lns"]),
    W_comb_lnb_(model["decoder_W_comb_att_lnb"])
{}

Weights::DecSoftmax::DecSoftmax(const NpzConverter& model)
  : W1_(model["ff_logit_lstm_W"]),
    B1_(model("ff_logit_lstm_b", true)),
    W2_(model["ff_logit_prev_W"]),
    B2_(model("ff_logit_prev_b", true)),
    W3_(model["ff_logit_ctx_W"]),
    B3_(model("ff_logit_ctx_b", true)),
    W4_(model.getFirstOfMany({std::make_pair(std::string("ff_logit_W"), false),
                              std::make_pair(std::string("Wemb_dec"), true)})),
    B4_(model("ff_logit_b", true)),
    lns_1_(model["ff_logit_lstm_ln_s"]),
    lns_2_(model["ff_logit_prev_ln_s"]),
    lns_3_(model["ff_logit_ctx_ln_s"]),
    lnb_1_(model["ff_logit_lstm_ln_b"]),
    lnb_2_(model["ff_logit_prev_ln_b"]),
    lnb_3_(model["ff_logit_ctx_ln_b"])
{}

//////////////////////////////////////////////////////////////////////////////

Weights::Weights(const NpzConverter& model, size_t)
  : encEmbeddings_(model, "Wemb"),
    decEmbeddings_(model, std::vector<std::pair<std::string, bool>>(
          {std::make_pair(std::string("Wemb_dec"), false),
           std::make_pair(std::string("Wemb"), false)})),
    encForwardGRU_(model, "encoder_", {"W", "b", "U", "Wx", "bx", "Ux", "W_lns", "W_lnb", "Wx_lns",
                                       "Wx_lnb", "U_lns", "U_lnb", "Ux_lns", "Ux_lnb" }),
    encBackwardGRU_(model, "encoder_r_", {"W", "b", "U", "Wx", "bx", "Ux", "W_lns", "W_lnb",
                                          "Wx_lns", "Wx_lnb", "U_lns", "U_lnb", "Ux_lns", "Ux_lnb" }),
    decInit_(model),
    decGru1_(model, "decoder_", {"W", "b", "U", "Wx", "bx", "Ux", "W_lns", "W_lnb", "Wx_lns",
                                 "Wx_lnb", "U_lns", "U_lnb", "Ux_lns", "Ux_lnb" }),
    decGru2_(model, "decoder_", {"Wc", "U_nl", "b_nl", "Wcx", "Ux_nl", "bx_nl", "Wc_lns", "Wc_lnb",
                                 "Wcx_lns", "Wcx_lnb", "U_nl_lns", "U_nl_lnb", "Ux_nl_lns",
                                 "Ux_nl_lnb"}),
    decAttention_(model),
    decSoftmax_(model),
    encForwardTransition_(model, Weights::Transition::TransitionType::Encoder, "encoder_"),
    encBackwardTransition_(model,Weights::Transition::TransitionType::Encoder, "encoder_r_"),
    decTransition_(model, Weights::Transition::TransitionType::Decoder, "decoder_", "_nl")
{}

}  // namespace Nematus
}  // namespace cpu
}  // namespace amunmt
