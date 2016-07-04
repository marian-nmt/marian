#include "model.h"

using namespace std;

Weights::EncEmbeddings::EncEmbeddings(const NpzConverter& model)
: E_(model["Wemb"])
{}

Weights::EncForwardGRU::EncForwardGRU(const NpzConverter& model)
: W_(model["encoder_W"]),
  B_(model("encoder_b", true)),
  U_(model["encoder_U"]),
  Wx_(model["encoder_Wx"]),
  Bx1_(model("encoder_bx", true)),
  Bx2_(Bx1_.Rows(), Bx1_.Cols(), 0.0),
  Ux_(model["encoder_Ux"])
{ }

Weights::EncBackwardGRU::EncBackwardGRU(const NpzConverter& model)
: W_(model["encoder_r_W"]),
  B_(model("encoder_r_b", true)),
  U_(model["encoder_r_U"]),
  Wx_(model["encoder_r_Wx"]),
  Bx1_(model("encoder_r_bx", true)),
  Bx2_(Bx1_.Rows(), Bx1_.Cols(), 0.0),
  Ux_(model["encoder_r_Ux"])
{}

//////////////////////////////////////////////////////////////////////////////

Weights::DecEmbeddings::DecEmbeddings(const NpzConverter& model)
: E_(model["Wemb_dec"])
{}

Weights::DecInit::DecInit(const NpzConverter& model)
: Wi_(model["ff_state_W"]),
  Bi_(model("ff_state_b", true))
{}

Weights::DecGRU1::DecGRU1(const NpzConverter& model)
: W_(model["decoder_W"]),
  B_(model("decoder_b", true)),
  U_(model["decoder_U"]),
  Wx_(model["decoder_Wx"]),
  Bx1_(model("decoder_bx", true)),
  Bx2_(Bx1_.Rows(), Bx1_.Cols(), 0.0),
  Ux_(model["decoder_Ux"])
{}

Weights::DecGRU2::DecGRU2(const NpzConverter& model)
: W_(model["decoder_Wc"]),
  B_(model("decoder_b_nl", true)),
  U_(model["decoder_U_nl"]),
  Wx_(model["decoder_Wcx"]),
  Bx2_(model("decoder_bx_nl", true)),
  Bx1_(Bx2_.Rows(), Bx2_.Cols(), 0.0),
  Ux_(model["decoder_Ux_nl"])
{}

Weights::DecAttention::DecAttention(const NpzConverter& model)
: V_(model("decoder_U_att", true)),
W_(model["decoder_W_comb_att"]),
B_(model("decoder_b_att", true)),
U_(model["decoder_Wc_att"]),
C_(model["decoder_c_tt"]) // scalar?
{}

Weights::DecSoftmax::DecSoftmax(const NpzConverter& model)
: W1_(model["ff_logit_lstm_W"]),
  B1_(model("ff_logit_lstm_b", true)),
  W2_(model["ff_logit_prev_W"]),
  B2_(model("ff_logit_prev_b", true)),
  W3_(model["ff_logit_ctx_W"]),
  B3_(model("ff_logit_ctx_b", true)),
  W4_(model["ff_logit_W"]),
  B4_(model("ff_logit_b", true))
{}

//////////////////////////////////////////////////////////////////////////////

Weights::Weights(const NpzConverter& model, size_t device)
: encEmbeddings_(model),
encForwardGRU_(model),
encBackwardGRU_(model),
decEmbeddings_(model),
decInit_(model),
decGru1_(model),
decGru2_(model),
decAttention_(model),
decSoftmax_(model),
device_(device)
{
	cerr << *this << endl;
}
