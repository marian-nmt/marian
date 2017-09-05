#pragma once

#include <map>
#include <string>
#include <yaml-cpp/yaml.h>

#include "gpu/mblas/matrix.h"
#include "gpu/npz_converter.h"

namespace amunmt {
namespace GPU {

struct Weights {

  //////////////////////////////////////////////////////////////////////////////

  struct EncEmbeddings {
    EncEmbeddings(const EncEmbeddings&) = delete;

    EncEmbeddings(const NpzConverter& model)
    : E_(model.get("Wemb", true))
    {}

    const std::shared_ptr<mblas::Matrix> E_;
  };

  struct EncForwardGRU {
    EncForwardGRU(const EncForwardGRU&) = delete;

    EncForwardGRU(const NpzConverter& model)
    : W_(model.get("encoder_W", true)),
      B_(model.get("encoder_b", true, true)),
      U_(model.get("encoder_U", true)),
      Wx_(model.get("encoder_Wx", true)),
      Bx1_(model.get("encoder_bx", true, true)),
      Bx2_(new mblas::Matrix(Bx1_->dim(0), Bx1_->dim(1), Bx1_->dim(2), Bx1_->dim(3), true)),
      Ux_(model.get("encoder_Ux", true)),
      Gamma_1_(model.get("encoder_gamma1", false)),
      Gamma_2_(model.get("encoder_gamma2", false))
    { }

    const std::shared_ptr<mblas::Matrix> W_;
    const std::shared_ptr<mblas::Matrix> B_;
    const std::shared_ptr<mblas::Matrix> U_;
    const std::shared_ptr<mblas::Matrix> Wx_;
    const std::shared_ptr<mblas::Matrix> Bx1_;
    const std::shared_ptr<mblas::Matrix> Bx2_;
    const std::shared_ptr<mblas::Matrix> Ux_;
    const std::shared_ptr<mblas::Matrix> Gamma_1_;
    const std::shared_ptr<mblas::Matrix> Gamma_2_;
  };

  struct EncBackwardGRU {
    EncBackwardGRU(const EncBackwardGRU&) = delete;

    EncBackwardGRU(const NpzConverter& model)
    : W_(model.get("encoder_r_W", true)),
      B_(model.get("encoder_r_b", true, true)),
      U_(model.get("encoder_r_U", true)),
      Wx_(model.get("encoder_r_Wx", true)),
      Bx1_(model.get("encoder_r_bx", true, true)),
      Bx2_(new mblas::Matrix( Bx1_->dim(0), Bx1_->dim(1), Bx1_->dim(2), Bx1_->dim(3), true)),
      Ux_(model.get("encoder_r_Ux", true)),
      Gamma_1_(model.get("encoder_r_gamma1", false)),
      Gamma_2_(model.get("encoder_r_gamma2", false))
    {}

    const std::shared_ptr<mblas::Matrix> W_;
    const std::shared_ptr<mblas::Matrix> B_;
    const std::shared_ptr<mblas::Matrix> U_;
    const std::shared_ptr<mblas::Matrix> Wx_;
    const std::shared_ptr<mblas::Matrix> Bx1_;
    const std::shared_ptr<mblas::Matrix> Bx2_;
    const std::shared_ptr<mblas::Matrix> Ux_;
    const std::shared_ptr<mblas::Matrix> Gamma_1_;
    const std::shared_ptr<mblas::Matrix> Gamma_2_;
  };

  struct EncForwardLSTM {
    EncForwardLSTM(const EncForwardLSTM&) = delete;

    EncForwardLSTM(const NpzConverter& model)
    : W_(model.get("encoder_W", true)),
      B_(model.get("encoder_b", true, true)),
      U_(model.get("encoder_U", true)),
      Wx_(model.get("encoder_Wx", true)),
      Bx1_(model.get("encoder_bx", true, true)),
      Bx2_(new mblas::Matrix(Bx1_->dim(0), Bx1_->dim(1), Bx1_->dim(2), Bx1_->dim(3), true)),
      Ux_(model.get("encoder_Ux", true)),
      Gamma_1_(model.get("encoder_gamma1", false)),
      Gamma_2_(model.get("encoder_gamma2", false))
    { }

    const std::shared_ptr<mblas::Matrix> W_;
    const std::shared_ptr<mblas::Matrix> B_;
    const std::shared_ptr<mblas::Matrix> U_;
    const std::shared_ptr<mblas::Matrix> Wx_;
    const std::shared_ptr<mblas::Matrix> Bx1_;
    const std::shared_ptr<mblas::Matrix> Bx2_;
    const std::shared_ptr<mblas::Matrix> Ux_;
    const std::shared_ptr<mblas::Matrix> Gamma_1_;
    const std::shared_ptr<mblas::Matrix> Gamma_2_;
  };

  struct EncBackwardLSTM {
    EncBackwardLSTM(const EncBackwardLSTM&) = delete;

    EncBackwardLSTM(const NpzConverter& model)
    : W_(model.get("encoder_r_W", true)),
      B_(model.get("encoder_r_b", true, true)),
      U_(model.get("encoder_r_U", true)),
      Wx_(model.get("encoder_r_Wx", true)),
      Bx1_(model.get("encoder_r_bx", true, true)),
      Bx2_(new mblas::Matrix( Bx1_->dim(0), Bx1_->dim(1), Bx1_->dim(2), Bx1_->dim(3), true)),
      Ux_(model.get("encoder_r_Ux", true)),
      Gamma_1_(model.get("encoder_r_gamma1", false)),
      Gamma_2_(model.get("encoder_r_gamma2", false))
    {}

    const std::shared_ptr<mblas::Matrix> W_;
    const std::shared_ptr<mblas::Matrix> B_;
    const std::shared_ptr<mblas::Matrix> U_;
    const std::shared_ptr<mblas::Matrix> Wx_;
    const std::shared_ptr<mblas::Matrix> Bx1_;
    const std::shared_ptr<mblas::Matrix> Bx2_;
    const std::shared_ptr<mblas::Matrix> Ux_;
    const std::shared_ptr<mblas::Matrix> Gamma_1_;
    const std::shared_ptr<mblas::Matrix> Gamma_2_;
  };

  //////////////////////////////////////////////////////////////////////////////

  struct DecEmbeddings {
    DecEmbeddings(const DecEmbeddings&) = delete;

    DecEmbeddings(const NpzConverter& model)
    : E_(model.getFirstOfMany({std::make_pair("Wemb_dec", false),
                               std::make_pair("Wemb", false)}, true))
    {}

    const std::shared_ptr<mblas::Matrix> E_;
  };

  struct DecInit {
    DecInit(const DecInit&) = delete;

    DecInit(const NpzConverter& model)
    : Wi_(model.get("ff_state_W", true)),
      Bi_(model.get("ff_state_b", true, true)),
      Gamma_(model.get("ff_state_gamma", false))
    {}

    const std::shared_ptr<mblas::Matrix> Wi_;
    const std::shared_ptr<mblas::Matrix> Bi_;
    const std::shared_ptr<mblas::Matrix> Gamma_;
  };

  struct DecGRU1 {
    DecGRU1(const DecGRU1&) = delete;

    DecGRU1(const NpzConverter& model)
    : W_(model.get("decoder_W", true)),
      B_(model.get("decoder_b", true, true)),
      U_(model.get("decoder_U", true)),
      Wx_(model.get("decoder_Wx", true)),
      Bx1_(model.get("decoder_bx", true, true)),
      Bx2_(new mblas::Matrix(Bx1_->dim(0), Bx1_->dim(1), Bx1_->dim(2), Bx1_->dim(3), true)),
      Ux_(model.get("decoder_Ux", true)),
      Gamma_1_(model.get("decoder_cell1_gamma1", false)),
      Gamma_2_(model.get("decoder_cell1_gamma2", false))
    {}

    const std::shared_ptr<mblas::Matrix> W_;
    const std::shared_ptr<mblas::Matrix> B_;
    const std::shared_ptr<mblas::Matrix> U_;
    const std::shared_ptr<mblas::Matrix> Wx_;
    const std::shared_ptr<mblas::Matrix> Bx1_;
    const std::shared_ptr<mblas::Matrix> Bx2_;
    const std::shared_ptr<mblas::Matrix> Ux_;
    const std::shared_ptr<mblas::Matrix> Gamma_1_;
    const std::shared_ptr<mblas::Matrix> Gamma_2_;
  };

  struct DecGRU2 {
    DecGRU2(const DecGRU2&) = delete;

    DecGRU2(const NpzConverter& model)
    : W_(model.get("decoder_Wc", true)),
      B_(model.get("decoder_b_nl", true, true)),
      U_(model.get("decoder_U_nl", true)),
      Wx_(model.get("decoder_Wcx", true)),
      Bx2_(model.get("decoder_bx_nl", true, true)),
      Bx1_(new mblas::Matrix(Bx2_->dim(0), Bx2_->dim(1), Bx2_->dim(2), Bx2_->dim(3), true)),
      Ux_(model.get("decoder_Ux_nl", true)),
      Gamma_1_(model.get("decoder_cell2_gamma1", false)),
      Gamma_2_(model.get("decoder_cell2_gamma2", false))
    {}

    const std::shared_ptr<mblas::Matrix> W_;
    const std::shared_ptr<mblas::Matrix> B_;
    const std::shared_ptr<mblas::Matrix> U_;
    const std::shared_ptr<mblas::Matrix> Wx_;
    const std::shared_ptr<mblas::Matrix> Bx2_;
    const std::shared_ptr<mblas::Matrix> Bx1_;
    const std::shared_ptr<mblas::Matrix> Ux_;
    const std::shared_ptr<mblas::Matrix> Gamma_1_;
    const std::shared_ptr<mblas::Matrix> Gamma_2_;
  };

  struct DecLSTM1 {
    DecLSTM1(const DecLSTM1&) = delete;

    DecLSTM1(const NpzConverter& model)
    : W_(model.get("decoder_W", true)),
      B_(model.get("decoder_b", true, true)),
      U_(model.get("decoder_U", true)),
      Wx_(model.get("decoder_Wx", true)),
      Bx1_(model.get("decoder_bx", true, true)),
      Bx2_(new mblas::Matrix(Bx1_->dim(0), Bx1_->dim(1), Bx1_->dim(2), Bx1_->dim(3), true)),
      Ux_(model.get("decoder_Ux", true)),
      Gamma_1_(model.get("decoder_cell1_gamma1", false)),
      Gamma_2_(model.get("decoder_cell1_gamma2", false))
    {}

    const std::shared_ptr<mblas::Matrix> W_;
    const std::shared_ptr<mblas::Matrix> B_;
    const std::shared_ptr<mblas::Matrix> U_;
    const std::shared_ptr<mblas::Matrix> Wx_;
    const std::shared_ptr<mblas::Matrix> Bx1_;
    const std::shared_ptr<mblas::Matrix> Bx2_;
    const std::shared_ptr<mblas::Matrix> Ux_;
    const std::shared_ptr<mblas::Matrix> Gamma_1_;
    const std::shared_ptr<mblas::Matrix> Gamma_2_;
  };

  struct DecLSTM2 {
    DecLSTM2(const DecLSTM2&) = delete;

    DecLSTM2(const NpzConverter& model)
    : W_(model.get("decoder_Wc", true)),
      B_(model.get("decoder_b_nl", true, true)),
      U_(model.get("decoder_U_nl", true)),
      Wx_(model.get("decoder_Wcx", true)),
      Bx2_(model.get("decoder_bx_nl", true, true)),
      Bx1_(new mblas::Matrix(Bx2_->dim(0), Bx2_->dim(1), Bx2_->dim(2), Bx2_->dim(3), true)),
      Ux_(model.get("decoder_Ux_nl", true)),
      Gamma_1_(model.get("decoder_cell2_gamma1", false)),
      Gamma_2_(model.get("decoder_cell2_gamma2", false))
    {}

    const std::shared_ptr<mblas::Matrix> W_;
    const std::shared_ptr<mblas::Matrix> B_;
    const std::shared_ptr<mblas::Matrix> U_;
    const std::shared_ptr<mblas::Matrix> Wx_;
    const std::shared_ptr<mblas::Matrix> Bx2_;
    const std::shared_ptr<mblas::Matrix> Bx1_;
    const std::shared_ptr<mblas::Matrix> Ux_;
    const std::shared_ptr<mblas::Matrix> Gamma_1_;
    const std::shared_ptr<mblas::Matrix> Gamma_2_;
  };

  struct DecAlignment {
    DecAlignment(const DecAlignment&) = delete;

    DecAlignment(const NpzConverter& model)
    : V_(model.get("decoder_U_att", true, true)),
      W_(model.get("decoder_W_comb_att", true)),
      B_(model.get("decoder_b_att", true, true)),
      U_(model.get("decoder_Wc_att", true)),
      C_(model.get("decoder_c_tt", true)), // scalar?
      Gamma_1_(model.get("decoder_att_gamma1", false)),
      Gamma_2_(model.get("decoder_att_gamma2", false))
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
    : W1_(model.get("ff_logit_lstm_W", true)),
      B1_(model.get("ff_logit_lstm_b", true, true)),
      W2_(model.get("ff_logit_prev_W", true)),
      B2_(model.get("ff_logit_prev_b", true, true)),
      W3_(model.get("ff_logit_ctx_W", true)),
      B3_(model.get("ff_logit_ctx_b", true, true)),
      W4_(model.getFirstOfMany({std::make_pair(std::string("ff_logit_W"), false),
                                std::make_pair(std::string("Wemb_dec"), true)}, true)),
      B4_(model.get("ff_logit_b", true, true)),
      Gamma_0_(model.get("ff_logit_l1_gamma0", false)),
      Gamma_1_(model.get("ff_logit_l1_gamma1", false)),
      Gamma_2_(model.get("ff_logit_l1_gamma2", false))
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

  Weights(const std::string& npzFile, const YAML::Node& config,  size_t device)
  : Weights(NpzConverter(npzFile), config, device)
  {}

  Weights(const NpzConverter& model, const YAML::Node& config, size_t device)
  : encEmbeddings_(model),
    decEmbeddings_(model),
    decInit_(model),
    decAlignment_(model),
    decSoftmax_(model),
    device_(device)
    {

      if (config["enc-cell"] && config["enc-cell"].as<std::string>() == "lstm") {
        encForwardLSTM_ = std::shared_ptr<EncForwardLSTM>(new EncForwardLSTM(model));
        encBackwardLSTM_ = std::shared_ptr<EncBackwardLSTM>(new EncBackwardLSTM(model));
      } else {
        encForwardGRU_ = std::shared_ptr<EncForwardGRU>(new EncForwardGRU(model));
        encBackwardGRU_ = std::shared_ptr<EncBackwardGRU>(new EncBackwardGRU(model));
      }
      if (config["dec-cell"] && config["dec-cell"].as<std::string>() == "lstm") {
        decLSTM1_ = std::shared_ptr<DecLSTM1>(new DecLSTM1(model));
        decLSTM2_ = std::shared_ptr<DecLSTM2>(new DecLSTM2(model));
      } else {
        decGru1_ = std::shared_ptr<DecGRU1>(new DecGRU1(model));
        decGru2_ = std::shared_ptr<DecGRU2>(new DecGRU2(model));
      }
    }

  Weights(const Weights&) = delete;

  size_t GetDevice() {
    return device_;
  }

  const EncEmbeddings encEmbeddings_;
  const DecEmbeddings decEmbeddings_;
  std::shared_ptr<EncForwardGRU> encForwardGRU_;
  std::shared_ptr<EncBackwardGRU> encBackwardGRU_;
  std::shared_ptr<EncForwardLSTM> encForwardLSTM_;
  std::shared_ptr<EncBackwardLSTM> encBackwardLSTM_;
  const DecInit decInit_;
  std::shared_ptr<DecGRU1> decGru1_;
  std::shared_ptr<DecGRU2> decGru2_;
  std::shared_ptr<DecLSTM1> decLSTM1_;
  std::shared_ptr<DecLSTM2> decLSTM2_;
  const DecAlignment decAlignment_;
  const DecSoftmax decSoftmax_;

  const size_t device_;
};

}
}


