#pragma once
#include <string>
#include "matrix.h"
#include "npz_converter.h"

namespace amunmt {
namespace FPGA {

struct Weights {

  //////////////////////////////////////////////////////////////////////////////
  struct EncEmbeddings {

    EncEmbeddings(const cl_context &context, const cl_device_id &device, const NpzConverter& model)
    : E_(model.GetMatrix(context, device, "Wemb"))
    {
      //std::cerr << "E_=" << E_.Debug() << std::endl;
    }

    const mblas::Matrix E_;
  };

  struct EncForwardGRU {
    EncForwardGRU(const cl_context &context, const cl_device_id &device, const NpzConverter& model)
    : W_(model.GetMatrix(context, device, "encoder_W")),
      B_(model.GetMatrix(context, device, "encoder_b", true)),
      U_(model.GetMatrix(context, device, "encoder_U")),
      Wx_(model.GetMatrix(context, device, "encoder_Wx")),
      Bx1_(model.GetMatrix(context, device, "encoder_bx", true)),
      Bx2_(context, device, Bx1_.dim(0), Bx1_.dim(1), true),
      Ux_(model.GetMatrix(context, device, "encoder_Ux"))
    { }

    const mblas::Matrix W_;
    const mblas::Matrix B_;
    const mblas::Matrix U_;
    const mblas::Matrix Wx_;
    const mblas::Matrix Bx1_;
    const mblas::Matrix Bx2_;
    const mblas::Matrix Ux_;
  };

  struct EncBackwardGRU {
    EncBackwardGRU(const cl_context &context, const cl_device_id &device, const NpzConverter& model)
    : W_(model.GetMatrix(context, device, "encoder_r_W")),
      B_(model.GetMatrix(context, device, "encoder_r_b", true)),
      U_(model.GetMatrix(context, device, "encoder_r_U")),
      Wx_(model.GetMatrix(context, device, "encoder_r_Wx")),
      Bx1_(model.GetMatrix(context, device, "encoder_r_bx", true)),
      Bx2_(context, device, Bx1_.dim(0), Bx1_.dim(1), true),
      Ux_(model.GetMatrix(context, device, "encoder_r_Ux"))
    {}

    const mblas::Matrix W_;
    const mblas::Matrix B_;
    const mblas::Matrix U_;
    const mblas::Matrix Wx_;
    const mblas::Matrix Bx1_;
    const mblas::Matrix Bx2_;
    const mblas::Matrix Ux_;
  };

  //////////////////////////////////////////////////////////////////////////////
  struct DecEmbeddings {
    DecEmbeddings(const cl_context &context, const cl_device_id &device, const NpzConverter& model)
    : E_(model.GetMatrix(context, device, "Wemb_dec"))
    {}

    const mblas::Matrix E_;
  };

  struct DecInit {
    DecInit(const cl_context &context, const cl_device_id &device, const NpzConverter& model)
    : Wi_(model.GetMatrix(context, device, "ff_state_W")),
      Bi_(model.GetMatrix(context, device, "ff_state_b", true))
    {}

    const mblas::Matrix Wi_;
    const mblas::Matrix Bi_;
  };

  struct DecGRU1 {
    DecGRU1(const cl_context &context, const cl_device_id &device, const NpzConverter& model)
    : W_(model.GetMatrix(context, device, "decoder_W")),
      B_(model.GetMatrix(context, device, "decoder_b", true)),
      U_(model.GetMatrix(context, device, "decoder_U")),
      Wx_(model.GetMatrix(context, device, "decoder_Wx")),
      Bx1_(model.GetMatrix(context, device, "decoder_bx", true)),
      Bx2_(context, device, Bx1_.dim(0), Bx1_.dim(1), true),
      Ux_(model.GetMatrix(context, device, "decoder_Ux"))
    {}

    const mblas::Matrix W_;
    const mblas::Matrix B_;
    const mblas::Matrix U_;
    const mblas::Matrix Wx_;
    const mblas::Matrix Bx1_;
    const mblas::Matrix Bx2_;
    const mblas::Matrix Ux_;
  };

  struct DecGRU2 {
    DecGRU2(const cl_context &context, const cl_device_id &device, const NpzConverter& model)
    : W_(model.GetMatrix(context, device, "decoder_Wc")),
      B_(model.GetMatrix(context, device, "decoder_b_nl", true)),
      U_(model.GetMatrix(context, device, "decoder_U_nl")),
      Wx_(model.GetMatrix(context, device, "decoder_Wcx")),
      Bx2_(model.GetMatrix(context, device, "decoder_bx_nl", true)),
      Bx1_(context, device, Bx2_.dim(0), Bx2_.dim(1), true),
      Ux_(model.GetMatrix(context, device, "decoder_Ux_nl"))
    {}

    const mblas::Matrix W_;
    const mblas::Matrix B_;
    const mblas::Matrix U_;
    const mblas::Matrix Wx_;
    const mblas::Matrix Bx2_;
    const mblas::Matrix Bx1_;
    const mblas::Matrix Ux_;
  };

  struct DecAlignment {
    DecAlignment(const cl_context &context, const cl_device_id &device, const NpzConverter& model)
    : V_(model.GetMatrix(context, device, "decoder_U_att", true)),
      W_(model.GetMatrix(context, device, "decoder_W_comb_att")),
      B_(model.GetMatrix(context, device, "decoder_b_att", true)),
      U_(model.GetMatrix(context, device, "decoder_Wc_att")),
      C_(model.GetMatrix(context, device, "decoder_c_tt")) // scalar?
    {}

    const mblas::Matrix V_;
    const mblas::Matrix W_;
    const mblas::Matrix B_;
    const mblas::Matrix U_;
    const mblas::Matrix C_;
  };

  struct DecSoftmax {
    DecSoftmax(const cl_context &context, const cl_device_id &device, const NpzConverter& model)
    : W1_(model.GetMatrix(context, device, "ff_logit_lstm_W")),
      B1_(model.GetMatrix(context, device, "ff_logit_lstm_b", true)),
      W2_(model.GetMatrix(context, device, "ff_logit_prev_W")),
      B2_(model.GetMatrix(context, device, "ff_logit_prev_b", true)),
      W3_(model.GetMatrix(context, device, "ff_logit_ctx_W")),
      B3_(model.GetMatrix(context, device, "ff_logit_ctx_b", true)),
      W4_(model.GetMatrix(context, device, "ff_logit_W")),
      B4_(model.GetMatrix(context, device, "ff_logit_b", true))
    {}

    const mblas::Matrix W1_;
    const mblas::Matrix B1_;
    const mblas::Matrix W2_;
    const mblas::Matrix B2_;
    const mblas::Matrix W3_;
    const mblas::Matrix B3_;
    const mblas::Matrix W4_;
    const mblas::Matrix B4_;
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

