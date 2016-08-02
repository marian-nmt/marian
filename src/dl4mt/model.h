#pragma once

#include <map>
#include <string>

#include "mblas/matrix.h"
#include "npz_converter.h"

std::string ENC_NAME(const std::string prefix, const std::string suffix, const size_t index);

struct Weights {
  using TMatrix = mblas::Matrix;

  //////////////////////////////////////////////////////////////////////////////

  struct EncEmbeddings {
    EncEmbeddings(const NpzConverter& model, size_t index=0)
    : E_(model[ENC_NAME("Wemb", "",index)])
    {}

    const TMatrix E_;
  };

  struct EncForwardGRU {
    EncForwardGRU(const NpzConverter& model, size_t index=0)
    : W_(model[ENC_NAME("encoder", "_W", index)]),
      B_(model(ENC_NAME("encoder", "_b", index), true)),
      U_(model[ENC_NAME("encoder", "_U", index)]),
      Wx_(model[ENC_NAME("encoder", "_Wx", index)]),
      Bx1_(model(ENC_NAME("encoder", "_bx", index), true)),
      Bx2_(Bx1_.Rows(), Bx1_.Cols(), 0.0),
      Ux_(model[ENC_NAME("encoder", "_Ux", index)])
    {}

    const TMatrix W_;
    const TMatrix B_;
    const TMatrix U_;
    const TMatrix Wx_;
    const TMatrix Bx1_;
    const TMatrix Bx2_;
    const TMatrix Ux_;
  };

  struct EncBackwardGRU {
    EncBackwardGRU(const NpzConverter& model, size_t index=0)
    : W_(model[ENC_NAME("encoder_r", "_W", index)]),
      B_(model(ENC_NAME("encoder_r", "_b", index), true)),
      U_(model[ENC_NAME("encoder_r", "_U", index)]),
      Wx_(model[ENC_NAME("encoder_r", "_Wx", index)]),
      Bx1_(model(ENC_NAME("encoder_r", "_bx", index), true)),
      Bx2_(Bx1_.Rows(), Bx1_.Cols(), 0.0),
      Ux_(model[ENC_NAME("encoder_r", "_Ux", index)])
    {}

    const TMatrix W_;
    const TMatrix B_;
    const TMatrix U_;
    const TMatrix Wx_;
    const TMatrix Bx1_;
    const TMatrix Bx2_;
    const TMatrix Ux_;
  };

  //////////////////////////////////////////////////////////////////////////////

  struct DecEmbeddings {
    DecEmbeddings(const NpzConverter& model)
    : E_(model["Wemb_dec"])
    {}

    const TMatrix E_;
  };

  struct DecInit {
    DecInit(const NpzConverter& model)
    : Wi_(model["ff_state_W"]),
      Bi_(model("ff_state_b", true))
    {}

    const TMatrix Wi_;
    const TMatrix Bi_;
  };

  struct DecGRU1 {
    DecGRU1(const NpzConverter& model)
    : W_(model["decoder_W"]),
      B_(model("decoder_b", true)),
      U_(model["decoder_U"]),
      Wx_(model["decoder_Wx"]),
      Bx1_(model("decoder_bx", true)),
      Bx2_(Bx1_.Rows(), Bx1_.Cols(), 0.0),
      Ux_(model["decoder_Ux"])
    {}

    const TMatrix W_;
    const TMatrix B_;
    const TMatrix U_;
    const TMatrix Wx_;
    const TMatrix Bx1_;
    const TMatrix Bx2_;
    const TMatrix Ux_;
  };

  struct DecGRU2 {
    DecGRU2(const NpzConverter& model)
    : W_(model["decoder_Wc"]),
      B_(model("decoder_b_nl", true)),
      U_(model["decoder_U_nl"]),
      Wx_(model["decoder_Wcx"]),
      Bx2_(model("decoder_bx_nl", true)),
      Bx1_(Bx2_.Rows(), Bx2_.Cols(), 0.0),
      Ux_(model["decoder_Ux_nl"])
    {}

    const TMatrix W_;
    const TMatrix B_;
    const TMatrix U_;
    const TMatrix Wx_;
    const TMatrix Bx2_;
    const TMatrix Bx1_;
    const TMatrix Ux_;
  };

  struct DecAlignment {
    DecAlignment(const NpzConverter& model, const size_t index=0)
    : V_(model(ENC_NAME("decoder_U_att", "", index), true)),
      W_(model[ENC_NAME("decoder_W_comb_att", "", index)]),
      B_(model(ENC_NAME("decoder_b_att", "", index), true)),
      U_(model[ENC_NAME("decoder_Wc_att", "", index)]),
      C_(model[ENC_NAME("decoder_c_tt", "", index)]) // scalar?
    {}

    const TMatrix V_;
    const TMatrix W_;
    const TMatrix B_;
    const TMatrix U_;
    const TMatrix C_;
  };

  struct DecSoftmax {
    DecSoftmax(const NpzConverter& model)
    : W1_(model["ff_logit_lstm_W"]),
      B1_(model("ff_logit_lstm_b", true)),
      W2_(model["ff_logit_prev_W"]),
      B2_(model("ff_logit_prev_b", true)),
      W3_(model["ff_logit_ctx_W"]),
      B3_(model("ff_logit_ctx_b", true)),
      W4_(model["ff_logit_W"]),
      B4_(model("ff_logit_b", true))
    {}

    const TMatrix W1_;
    const TMatrix B1_;
    const TMatrix W2_;
    const TMatrix B2_;
    const TMatrix W3_;
    const TMatrix B3_;
    const TMatrix W4_;
    const TMatrix B4_;
  };

  Weights(const std::string& npzFile, size_t device = 0, size_t numEncoders=1)
  : Weights(NpzConverter(npzFile), device, numEncoders)
  {}

  Weights(const NpzConverter& model, size_t device = 0, size_t numEncoders=1)
    : decEmbeddings_(model),
      decInit_(model),
      decGru1_(model),
      decGru2_(model),
      decSoftmax_(model),
      device_(device) {
    for (size_t i = 0; i < numEncoders; ++i) {
      encEmbeddings_.emplace_back(new EncEmbeddings(model, i));
      encForwardGRU_.emplace_back(new EncForwardGRU(model, i));
      encBackwardGRU_.emplace_back(new EncBackwardGRU(model, i));
      decAlignment_.emplace_back(new DecAlignment(model, i));
    }
  }


  size_t GetDevice() {
    return device_;
  }

  std::vector<EncEmbeddings*> encEmbeddings_;
  std::vector<EncForwardGRU*> encForwardGRU_;
  std::vector<EncBackwardGRU*> encBackwardGRU_;
  const DecEmbeddings decEmbeddings_;
  const DecInit decInit_;
  const DecGRU1 decGru1_;
  const DecGRU2 decGru2_;
  std::vector<DecAlignment*> decAlignment_;
  const DecSoftmax decSoftmax_;

  const size_t device_;
};
