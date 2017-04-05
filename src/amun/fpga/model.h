#pragma once
#include <string>
#include "matrix.h"
#include "npz_converter.h"

namespace amunmt {
namespace FPGA {

struct Weights {

  //////////////////////////////////////////////////////////////////////////////
  struct EncEmbeddings {

    EncEmbeddings(const OpenCLInfo &openCLInfo, const NpzConverter& model)
    : E_(model.GetMatrix(openCLInfo, "Wemb"))
    {
      //std::cerr << "E_=" << E_.Debug() << std::endl;
    }

    const mblas::Matrix E_;
  };

  struct EncForwardGRU {
    EncForwardGRU(const OpenCLInfo &openCLInfo, const NpzConverter& model)
    : W_(model.GetMatrix(openCLInfo, "encoder_W")),
      B_(model.GetMatrix(openCLInfo, "encoder_b", true)),
      U_(model.GetMatrix(openCLInfo, "encoder_U")),
      Wx_(model.GetMatrix(openCLInfo, "encoder_Wx")),
      Bx1_(model.GetMatrix(openCLInfo, "encoder_bx", true)),
      Bx2_(openCLInfo, Bx1_.dim(0), Bx1_.dim(1), true),
      Ux_(model.GetMatrix(openCLInfo, "encoder_Ux")),
      Gamma_1_(model.GetMatrix(openCLInfo, "encoder_gamma1")),
      Gamma_2_(model.GetMatrix(openCLInfo, "encoder_gamma2"))
    { }

    const mblas::Matrix W_;
    const mblas::Matrix B_;
    const mblas::Matrix U_;
    const mblas::Matrix Wx_;
    const mblas::Matrix Bx1_;
    const mblas::Matrix Bx2_;
    const mblas::Matrix Ux_;
    const mblas::Matrix Gamma_1_;
    const mblas::Matrix Gamma_2_;
  };

  struct EncBackwardGRU {
    EncBackwardGRU(const OpenCLInfo &openCLInfo, const NpzConverter& model)
    : W_(model.GetMatrix(openCLInfo, "encoder_r_W")),
      B_(model.GetMatrix(openCLInfo, "encoder_r_b", true)),
      U_(model.GetMatrix(openCLInfo, "encoder_r_U")),
      Wx_(model.GetMatrix(openCLInfo, "encoder_r_Wx")),
      Bx1_(model.GetMatrix(openCLInfo, "encoder_r_bx", true)),
      Bx2_(openCLInfo, Bx1_.dim(0), Bx1_.dim(1), true),
      Ux_(model.GetMatrix(openCLInfo, "encoder_r_Ux")),
      Gamma_1_(model.GetMatrix(openCLInfo, "encoder_r_gamma1")),
      Gamma_2_(model.GetMatrix(openCLInfo, "encoder_r_gamma2"))
    {}

    const mblas::Matrix W_;
    const mblas::Matrix B_;
    const mblas::Matrix U_;
    const mblas::Matrix Wx_;
    const mblas::Matrix Bx1_;
    const mblas::Matrix Bx2_;
    const mblas::Matrix Ux_;
    const mblas::Matrix Gamma_1_;
    const mblas::Matrix Gamma_2_;
  };

  //////////////////////////////////////////////////////////////////////////////
  struct DecEmbeddings {
    DecEmbeddings(const OpenCLInfo &openCLInfo, const NpzConverter& model)
    : E_(model.GetMatrix(openCLInfo, "Wemb_dec"))
    {}

    const mblas::Matrix E_;
  };

  struct DecInit {
    DecInit(const OpenCLInfo &openCLInfo, const NpzConverter& model)
    : Wi_(model.GetMatrix(openCLInfo, "ff_state_W")),
      Bi_(model.GetMatrix(openCLInfo, "ff_state_b", true)),
      Gamma_(model.GetMatrix(openCLInfo, "ff_state_gamma"))
    {}

    const mblas::Matrix Wi_;
    const mblas::Matrix Bi_;
    const mblas::Matrix Gamma_;
  };

  struct DecGRU1 {
    DecGRU1(const OpenCLInfo &openCLInfo, const NpzConverter& model)
    : W_(model.GetMatrix(openCLInfo, "decoder_W")),
      B_(model.GetMatrix(openCLInfo, "decoder_b", true)),
      U_(model.GetMatrix(openCLInfo, "decoder_U")),
      Wx_(model.GetMatrix(openCLInfo, "decoder_Wx")),
      Bx1_(model.GetMatrix(openCLInfo, "decoder_bx", true)),
      Bx2_(openCLInfo, Bx1_.dim(0), Bx1_.dim(1), true),
      Ux_(model.GetMatrix(openCLInfo, "decoder_Ux")),
      Gamma_1_(model.GetMatrix(openCLInfo, "decoder_cell1_gamma1")),
      Gamma_2_(model.GetMatrix(openCLInfo, "decoder_cell1_gamma2"))
    {}

    const mblas::Matrix W_;
    const mblas::Matrix B_;
    const mblas::Matrix U_;
    const mblas::Matrix Wx_;
    const mblas::Matrix Bx1_;
    const mblas::Matrix Bx2_;
    const mblas::Matrix Ux_;
    const mblas::Matrix Gamma_1_;
    const mblas::Matrix Gamma_2_;
  };

  struct DecGRU2 {
    DecGRU2(const OpenCLInfo &openCLInfo, const NpzConverter& model)
    : W_(model.GetMatrix(openCLInfo, "decoder_Wc")),
      B_(model.GetMatrix(openCLInfo, "decoder_b_nl", true)),
      U_(model.GetMatrix(openCLInfo, "decoder_U_nl")),
      Wx_(model.GetMatrix(openCLInfo, "decoder_Wcx")),
      Bx2_(model.GetMatrix(openCLInfo, "decoder_bx_nl", true)),
      Bx1_(openCLInfo, Bx2_.dim(0), Bx2_.dim(1), true),
      Ux_(model.GetMatrix(openCLInfo, "decoder_Ux_nl")),
      Gamma_1_(model.GetMatrix(openCLInfo, "decoder_cell2_gamma1")),
      Gamma_2_(model.GetMatrix(openCLInfo, "decoder_cell2_gamma2"))
    {}

    const mblas::Matrix W_;
    const mblas::Matrix B_;
    const mblas::Matrix U_;
    const mblas::Matrix Wx_;
    const mblas::Matrix Bx2_;
    const mblas::Matrix Bx1_;
    const mblas::Matrix Ux_;
    const mblas::Matrix Gamma_1_;
    const mblas::Matrix Gamma_2_;
  };

  struct DecAlignment {
    DecAlignment(const OpenCLInfo &openCLInfo, const NpzConverter& model)
    : V_(model.GetMatrix(openCLInfo, "decoder_U_att", true)),
      W_(model.GetMatrix(openCLInfo, "decoder_W_comb_att")),
      B_(model.GetMatrix(openCLInfo, "decoder_b_att", true)),
      U_(model.GetMatrix(openCLInfo, "decoder_Wc_att")),
      C_(model.GetMatrix(openCLInfo, "decoder_c_tt")), // scalar?
      Gamma_1_(model.GetMatrix(openCLInfo, "decoder_att_gamma1")),
      Gamma_2_(model.GetMatrix(openCLInfo, "decoder_att_gamma2"))
    {}

    const mblas::Matrix V_;
    const mblas::Matrix W_;
    const mblas::Matrix B_;
    const mblas::Matrix U_;
    const mblas::Matrix C_;
    const mblas::Matrix Gamma_1_;
    const mblas::Matrix Gamma_2_;
  };

  struct DecSoftmax {
    DecSoftmax(const OpenCLInfo &openCLInfo, const NpzConverter& model)
    : W1_(model.GetMatrix(openCLInfo, "ff_logit_lstm_W")),
      B1_(model.GetMatrix(openCLInfo, "ff_logit_lstm_b", true)),
      W2_(model.GetMatrix(openCLInfo, "ff_logit_prev_W")),
      B2_(model.GetMatrix(openCLInfo, "ff_logit_prev_b", true)),
      W3_(model.GetMatrix(openCLInfo, "ff_logit_ctx_W")),
      B3_(model.GetMatrix(openCLInfo, "ff_logit_ctx_b", true)),
      W4_(model.GetMatrix(openCLInfo, "ff_logit_W")),
      B4_(model.GetMatrix(openCLInfo, "ff_logit_b", true)),
      Gamma_0_(model.GetMatrix(openCLInfo, "ff_logit_l1_gamma0")),
      Gamma_1_(model.GetMatrix(openCLInfo, "ff_logit_l1_gamma1")),
      Gamma_2_(model.GetMatrix(openCLInfo, "ff_logit_l1_gamma2"))
    {}

    const mblas::Matrix W1_;
    const mblas::Matrix B1_;
    const mblas::Matrix W2_;
    const mblas::Matrix B2_;
    const mblas::Matrix W3_;
    const mblas::Matrix B3_;
    const mblas::Matrix W4_;
    const mblas::Matrix B4_;
    const mblas::Matrix Gamma_0_;
    const mblas::Matrix Gamma_1_;
    const mblas::Matrix Gamma_2_;
  };

  //////////////////////////////////////////////////////////////////////////////
  Weights(const OpenCLInfo &openCLInfo, const std::string& npzFile);
  Weights(const OpenCLInfo &openCLInfo, const NpzConverter& model);

  EncEmbeddings encEmbeddings_;
  DecEmbeddings decEmbeddings_;
  EncForwardGRU encForwardGRU_;
  EncBackwardGRU encBackwardGRU_;
  DecInit decInit_;
  DecGRU1 decGru1_;
  DecGRU2 decGru2_;
  DecAlignment decAlignment_;
  DecSoftmax decSoftmax_;

  const OpenCLInfo &openCLInfo_;

};

}
}

