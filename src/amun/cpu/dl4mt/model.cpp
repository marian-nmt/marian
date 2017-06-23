#include "model.h"

using namespace std;

namespace amunmt {
namespace CPU {
namespace dl4mt {

Weights::Embeddings::Embeddings(const NpzConverter& model, const std::string &key)
  : E_(model[key])
{}

Weights::Embeddings::Embeddings(const NpzConverter& model, const std::vector<std::pair<std::string, bool>> keys)
  : E_(model.getFirstOfMany(keys))
{}

Weights::GRU::GRU(const NpzConverter& model, const std::vector<std::string> &keys)
  : W_(model[keys.at(0)]),
    B_(model(keys.at(1), true)),
    U_(model[keys.at(2)]),
    Wx_(model[keys.at(3)]),
    Bx1_(model(keys.at(4), true)),
    Bx2_(Bx1_.rows(), Bx1_.columns()),
    Ux_(model[keys.at(5)]),
    Gamma_1_(model[keys.at(6)]),
    Gamma_2_(model[keys.at(7)])
{
    const_cast<mblas::Matrix&>(Bx2_) = 0.0f;
}

//////////////////////////////////////////////////////////////////////////////

Weights::DecInit::DecInit(const NpzConverter& model)
  : Wi_(model["ff_state_W"]),
    Bi_(model("ff_state_b", true)),
    Gamma_(model["ff_state_gamma"])
{}

Weights::DecGRU2::DecGRU2(const NpzConverter& model)
: W_(model["decoder_Wc"]),
  B_(model("decoder_b_nl", true)),
  U_(model["decoder_U_nl"]),
  Wx_(model["decoder_Wcx"]),
  Bx2_(model("decoder_bx_nl", true)),
  Bx1_(Bx2_.rows(), Bx2_.columns()),
  Ux_(model["decoder_Ux_nl"]),
  Gamma_1_(model["decoder_cell2_gamma1"]),
  Gamma_2_(model["decoder_cell2_gamma2"])
{
    const_cast<mblas::Matrix&>(Bx1_) = 0.0f;
}

Weights::DecAttention::DecAttention(const NpzConverter& model)
: V_(model("decoder_U_att", true)),
  W_(model["decoder_W_comb_att"]),
  B_(model("decoder_b_att", true)),
  U_(model["decoder_Wc_att"]),
  C_(model["decoder_c_tt"]), // scalar?
  Gamma_1_(model["decoder_att_gamma1"]),
  Gamma_2_(model["decoder_att_gamma2"])
{}

Weights::DecSoftmax::DecSoftmax(const NpzConverter& model)
: W1_(model["ff_logit_lstm_W"]),
  B1_(model("ff_logit_lstm_b", true)),
  W2_(model["ff_logit_prev_W"]),
  B2_(model("ff_logit_prev_b", true)),
  W3_(model["ff_logit_ctx_W"]),
  B3_(model("ff_logit_ctx_b", true)),
  W4_(model.getFirstOfMany({std::pair<std::string, bool>(std::string("ff_logit_W"), false),
             std::make_pair(std::string("Wemb_dec"), true)})),
  B4_(model("ff_logit_b", true)),
  Gamma_0_(model["ff_logit_l1_gamma0"]),
  Gamma_1_(model["ff_logit_l1_gamma1"]),
  Gamma_2_(model["ff_logit_l1_gamma2"])
{}

//////////////////////////////////////////////////////////////////////////////

Weights::Weights(const NpzConverter& model, size_t)
: encEmbeddings_(model, "Wemb"),
  encForwardGRU_(model, {"encoder_W", "encoder_b", "encoder_U", "encoder_Wx", "encoder_bx",
                         "encoder_Ux", "encoder_gamma1", "encoder_gamma2"}),
  encBackwardGRU_(model, {"encoder_r_W", "encoder_r_b", "encoder_r_U", "encoder_r_Wx",
                          "encoder_r_bx", "encoder_r_Ux", "encoder_r_gamma1", "encoder_r_gamma2"}),
  decEmbeddings_(model, std::vector<std::pair<std::string, bool>>({std::make_pair(std::string("Wemb_dec"), false),
                         std::make_pair(std::string("Wemb"), false)})),
  decInit_(model),
  decGru1_(model, {"decoder_W", "decoder_b", "decoder_U", "decoder_Wx", "decoder_bx", "decoder_Ux",
                   "decoder_cell1_gamma1", "decoder_cell1_gamma2"}),
  decGru2_(model),
  decAttention_(model),
  decSoftmax_(model)
{}

}  // namespace dl4mt
}  // namespace cpu
}  // namespace amunmt
