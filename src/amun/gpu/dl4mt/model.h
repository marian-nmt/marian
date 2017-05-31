#pragma once

#include <map>
#include <string>

#include "gpu/mblas/matrix.h"
#include "gpu/npz_converter.h"

namespace amunmt {
namespace GPU {

struct Weights {

  //////////////////////////////////////////////////////////////////////////////

  struct EncEmbeddings {
    EncEmbeddings(const EncEmbeddings&) = delete;

    EncEmbeddings(const NpzConverter& model)
    : E_(model.getPtr("Wemb"))
    {}

    const std::shared_ptr<mblas::Matrix> E_;
  };

  struct EncForwardGRU {
    EncForwardGRU(const EncForwardGRU&) = delete;

    EncForwardGRU(const NpzConverter& model)
    : W_(model.getPtr("encoder_W")),
      B_(model.getPtr("encoder_b", true)),
      U_(model.getPtr("encoder_U")),
      Wx_(model.getPtr("encoder_Wx")),
      Bx1_(model.get("encoder_bx", true)),
      Bx2_(Bx1_.dim(0), Bx1_.dim(1), Bx1_.dim(2), Bx1_.dim(3), true),
      Ux_(model.getPtr("encoder_Ux")),
      Gamma_1_(model.getPtr("encoder_gamma1")),
      Gamma_2_(model.getPtr("encoder_gamma2"))
    { }

    const std::shared_ptr<mblas::Matrix> W_;
    const std::shared_ptr<mblas::Matrix> B_;
    const std::shared_ptr<mblas::Matrix> U_;
    const std::shared_ptr<mblas::Matrix> Wx_;
    const mblas::Matrix Bx1_;
    const mblas::Matrix Bx2_;
    const std::shared_ptr<mblas::Matrix> Ux_;
    const std::shared_ptr<mblas::Matrix> Gamma_1_;
    const std::shared_ptr<mblas::Matrix> Gamma_2_;
  };

  struct EncBackwardGRU {
    EncBackwardGRU(const EncBackwardGRU&) = delete;

    EncBackwardGRU(const NpzConverter& model)
    : W_(model.getPtr("encoder_r_W")),
      B_(model.getPtr("encoder_r_b", true)),
      U_(model.getPtr("encoder_r_U")),
      Wx_(model.getPtr("encoder_r_Wx")),
      Bx1_(model.get("encoder_r_bx", true)),
      Bx2_(Bx1_.dim(0), Bx1_.dim(1), Bx1_.dim(2), Bx1_.dim(3), true),
      Ux_(model.getPtr("encoder_r_Ux")),
      Gamma_1_(model.getPtr("encoder_r_gamma1")),
      Gamma_2_(model.getPtr("encoder_r_gamma2"))
    {}

    const std::shared_ptr<mblas::Matrix> W_;
    const std::shared_ptr<mblas::Matrix> B_;
    const std::shared_ptr<mblas::Matrix> U_;
    const std::shared_ptr<mblas::Matrix> Wx_;
    const mblas::Matrix Bx1_;
    const mblas::Matrix Bx2_;
    const std::shared_ptr<mblas::Matrix> Ux_;
    const std::shared_ptr<mblas::Matrix> Gamma_1_;
    const std::shared_ptr<mblas::Matrix> Gamma_2_;
  };

  //////////////////////////////////////////////////////////////////////////////

  struct DecEmbeddings {
    DecEmbeddings(const DecEmbeddings&) = delete;

    DecEmbeddings(const NpzConverter& model)
    : E_(model.getPtr("Wemb_dec"))
    {}

    const std::shared_ptr<mblas::Matrix> E_;
  };

  struct DecInit {
    DecInit(const DecInit&) = delete;

    DecInit(const NpzConverter& model)
    : Wi_(model.getPtr("ff_state_W")),
      Bi_(model.getPtr("ff_state_b", true)),
      Gamma_(model.getPtr("ff_state_gamma"))
    {}

    const std::shared_ptr<mblas::Matrix> Wi_;
    const std::shared_ptr<mblas::Matrix> Bi_;
    const std::shared_ptr<mblas::Matrix> Gamma_;
  };

  struct DecGRU1 {
    DecGRU1(const DecGRU1&) = delete;

    DecGRU1(const NpzConverter& model)
    : W_(model.getPtr("decoder_W")),
      B_(model.getPtr("decoder_b", true)),
      U_(model.getPtr("decoder_U")),
      Wx_(model.getPtr("decoder_Wx")),
      Bx1_(model.get("decoder_bx", true)),
      Bx2_(Bx1_.dim(0), Bx1_.dim(1), Bx1_.dim(2), Bx1_.dim(3), true),
      Ux_(model.getPtr("decoder_Ux")),
      Gamma_1_(model.getPtr("decoder_cell1_gamma1")),
      Gamma_2_(model.getPtr("decoder_cell1_gamma2"))
    {}

    const std::shared_ptr<mblas::Matrix> W_;
    const std::shared_ptr<mblas::Matrix> B_;
    const std::shared_ptr<mblas::Matrix> U_;
    const std::shared_ptr<mblas::Matrix> Wx_;
    const mblas::Matrix Bx1_;
    const mblas::Matrix Bx2_;
    const std::shared_ptr<mblas::Matrix> Ux_;
    const std::shared_ptr<mblas::Matrix> Gamma_1_;
    const std::shared_ptr<mblas::Matrix> Gamma_2_;
  };

  struct DecGRU2 {
    DecGRU2(const DecGRU2&) = delete;

    DecGRU2(const NpzConverter& model)
    : W_(model.getPtr("decoder_Wc")),
      B_(model.getPtr("decoder_b_nl", true)),
      U_(model.getPtr("decoder_U_nl")),
      Wx_(model.getPtr("decoder_Wcx")),
      Bx2_(model.get("decoder_bx_nl", true)),
      Bx1_(Bx2_.dim(0), Bx2_.dim(1), Bx2_.dim(2), Bx2_.dim(3), true),
      Ux_(model.getPtr("decoder_Ux_nl")),
      Gamma_1_(model.getPtr("decoder_cell2_gamma1")),
      Gamma_2_(model.getPtr("decoder_cell2_gamma2"))
    {}

    const std::shared_ptr<mblas::Matrix> W_;
    const std::shared_ptr<mblas::Matrix> B_;
    const std::shared_ptr<mblas::Matrix> U_;
    const std::shared_ptr<mblas::Matrix> Wx_;
    const mblas::Matrix Bx2_;
    const mblas::Matrix Bx1_;
    const std::shared_ptr<mblas::Matrix> Ux_;
    const std::shared_ptr<mblas::Matrix> Gamma_1_;
    const std::shared_ptr<mblas::Matrix> Gamma_2_;
  };

  struct DecAlignment {
    DecAlignment(const DecAlignment&) = delete;

    DecAlignment(const NpzConverter& model)
    : V_(model.getPtr("decoder_U_att", true)),
      W_(model.getPtr("decoder_W_comb_att")),
      B_(model.getPtr("decoder_b_att", true)),
      U_(model.getPtr("decoder_Wc_att")),
      C_(model.getPtr("decoder_c_tt")), // scalar?
      Gamma_1_(model.getPtr("decoder_att_gamma1")),
      Gamma_2_(model.getPtr("decoder_att_gamma2"))
    {}

    const std::shared_ptr<mblas::Matrix> V_;
    const std::shared_ptr<mblas::Matrix> W_;
    const std::shared_ptr<mblas::Matrix> B_;
    const std::shared_ptr<mblas::Matrix> U_;
    const std::shared_ptr<mblas::Matrix> C_;
    const std::shared_ptr<mblas::Matrix> Gamma_1_;
    const std::shared_ptr<mblas::Matrix> Gamma_2_;
  };

  struct DecSoftmax {
    DecSoftmax(const DecSoftmax&) = delete;

    DecSoftmax(const NpzConverter& model)
    : W1_(model.getPtr("ff_logit_lstm_W")),
      B1_(model.getPtr("ff_logit_lstm_b", true)),
      W2_(model.getPtr("ff_logit_prev_W")),
      B2_(model.getPtr("ff_logit_prev_b", true)),
      W3_(model.getPtr("ff_logit_ctx_W")),
      B3_(model.getPtr("ff_logit_ctx_b", true)),
      W4_(model.getPtr("ff_logit_W")),
      B4_(model.getPtr("ff_logit_b", true)),
      Gamma_0_(model.getPtr("ff_logit_l1_gamma0")),
      Gamma_1_(model.getPtr("ff_logit_l1_gamma1")),
      Gamma_2_(model.getPtr("ff_logit_l1_gamma2"))
    {}

    const std::shared_ptr<mblas::Matrix> W1_;
    const std::shared_ptr<mblas::Matrix> B1_;
    const std::shared_ptr<mblas::Matrix> W2_;
    const std::shared_ptr<mblas::Matrix> B2_;
    const std::shared_ptr<mblas::Matrix> W3_;
    const std::shared_ptr<mblas::Matrix> B3_;
    const std::shared_ptr<mblas::Matrix> W4_;
    const std::shared_ptr<mblas::Matrix> B4_;
    const std::shared_ptr<mblas::Matrix> Gamma_0_;
    const std::shared_ptr<mblas::Matrix> Gamma_1_;
    const std::shared_ptr<mblas::Matrix> Gamma_2_;
  };

  Weights(const std::string& npzFile, size_t device)
  : Weights(NpzConverter(npzFile), device)
  {}

  Weights(const NpzConverter& model, size_t device)
  : encEmbeddings_(model),
    encForwardGRU_(model),
    encBackwardGRU_(model),
    decEmbeddings_(model),
    decInit_(model),
    decGru1_(model),
    decGru2_(model),
    decAlignment_(model),
    decSoftmax_(model),
    device_(device)
    {}

  Weights(const Weights&) = delete;

  size_t GetDevice() {
    return device_;
  }

  const EncEmbeddings encEmbeddings_;
  const DecEmbeddings decEmbeddings_;
  const EncForwardGRU encForwardGRU_;
  const EncBackwardGRU encBackwardGRU_;
  const DecInit decInit_;
  const DecGRU1 decGru1_;
  const DecGRU2 decGru2_;
  const DecAlignment decAlignment_;
  const DecSoftmax decSoftmax_;

  const size_t device_;
};

}
}


