#include <sstream>
#include "model.h"
#include "gpu/mblas/tensor_functions.h"

using namespace std;

namespace {
  using namespace amunmt;
  using namespace GPU;
  using namespace mblas;
  shared_ptr<Tensor> merge (shared_ptr<Tensor> m1, shared_ptr<Tensor> m2) {
    if (m2->size()) {
      Transpose(*m1);
      Transpose(*m2);
      Concat(*m1, *m2);
      Transpose(*m1);
      Transpose(*m2);
    }
    return m1;
  }
}

namespace amunmt {
namespace GPU {

////////////////////////////////////////////////////////////////////////////////////////////////////
  Weights::EncEmbeddings::EncEmbeddings(const NpzConverter& model) {
  Es_.emplace_back(model.get("Wemb", true));

  for(int i=1; true; i++) {
    std::string factorKey = "Wemb" + std::to_string(i);
    std::shared_ptr<mblas::Tensor> factorEmb = model.get(factorKey, false);
    if (factorEmb->size() <= 0){
      break;
    }
    Es_.emplace_back(factorEmb);
  }
  }

  std::string Weights::EncEmbeddings::Debug(unsigned verbosity) const
  {
    stringstream strm;
    strm << "EncEmbeddings" << endl;
    return strm.str();
  }

////////////////////////////////////////////////////////////////////////////////////////////////////
Weights::EncForwardGRU::EncForwardGRU(const NpzConverter& model)
: W_(model.get("encoder_W", true)),
  B_(model.get("encoder_b", true, true)),
  U_(model.get("encoder_U", true)),
  Wx_(model.get("encoder_Wx", true)),
  Bx1_(model.get("encoder_bx", true, true)),
  Bx2_(new mblas::Tensor(Bx1_->dim(0), Bx1_->dim(1), Bx1_->dim(2), Bx1_->dim(3), true)),
  Ux_(model.get("encoder_Ux", true)),
  Gamma_1_(model.get("encoder_gamma1", false)),
  Gamma_2_(model.get("encoder_gamma2", false))
{ }

std::string Weights::EncForwardGRU::Debug(unsigned verbosity) const
{
  stringstream strm;
  strm << "W_=" << W_->Debug(verbosity) << endl;
  strm << "B_=" << B_->Debug(verbosity) << endl;
  strm << "U_=" << U_->Debug(verbosity) << endl;
  strm << "Wx_=" << Wx_->Debug(verbosity) << endl;
  strm << "Bx1_=" << Bx1_->Debug(verbosity) << endl;
  strm << "Bx2_=" << Bx2_->Debug(verbosity) << endl;
  strm << "Ux_=" << Ux_->Debug(verbosity) << endl;
  strm << "Gamma_1_=" << Gamma_1_->Debug(verbosity) << endl;
  strm << "Gamma_2_" << Gamma_2_->Debug(verbosity) << endl;

  return strm.str();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
Weights::EncForwardLSTM::EncForwardLSTM(const NpzConverter& model)
// matrix merging is done to be backwards-compatible with the original LSTM implementation in Amun
// we now use the same format used in Marian
// TODO: adapt to support Nematus LSTM models which use a similar format to Amun's original format
: W_(merge(model.get("encoder_W", true), model.get("encoder_Wx", false))),
  B_(merge(model.get("encoder_b", true, true), model.get("encoder_bx", false, true))),
  U_(merge(model.get("encoder_U", true), model.get("encoder_Ux", false))),
  Gamma_1_(model.get("encoder_gamma1", false)),
  Gamma_2_(model.get("encoder_gamma2", false))
{}

std::string Weights::EncForwardLSTM::Debug(unsigned verbosity) const
{
  stringstream strm;
  strm << "W_=" << W_->Debug(verbosity) << endl;
  strm << "B_=" << B_->Debug(verbosity) << endl;
  strm << "U_=" << U_->Debug(verbosity) << endl;
  strm << "Gamma_1_=" << Gamma_1_->Debug(verbosity) << endl;
  strm << "Gamma_2_" << Gamma_2_->Debug(verbosity) << endl;

  return strm.str();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
Weights::EncBackwardLSTM::EncBackwardLSTM(const NpzConverter& model)
// matrix merging is done to be backwards-compatible with the original LSTM implementation in Amun
// we now use the same format used in Marian
// TODO: adapt to support Nematus LSTM models which use a similar format to Amun's original format
: W_(merge(model.get("encoder_r_W", true), model.get("encoder_r_Wx", false))),
  B_(merge(model.get("encoder_r_b", true, true), model.get("encoder_r_bx", false, true))),
  U_(merge(model.get("encoder_r_U", true), model.get("encoder_r_Ux", false))),
  Gamma_1_(model.get("encoder_r_gamma1", false)),
  Gamma_2_(model.get("encoder_r_gamma2", false))
{}

std::string Weights::EncBackwardLSTM::Debug(unsigned verbosity) const
{
  stringstream strm;
  strm << "W_=" << W_->Debug(verbosity) << endl;
  strm << "B_=" << B_->Debug(verbosity) << endl;
  strm << "U_=" << U_->Debug(verbosity) << endl;
  strm << "Gamma_1_=" << Gamma_1_->Debug(verbosity) << endl;
  strm << "Gamma_2_" << Gamma_2_->Debug(verbosity) << endl;

  return strm.str();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
Weights::EncBackwardGRU::EncBackwardGRU(const NpzConverter& model)
: W_(model.get("encoder_r_W", true)),
  B_(model.get("encoder_r_b", true, true)),
  U_(model.get("encoder_r_U", true)),
  Wx_(model.get("encoder_r_Wx", true)),
  Bx1_(model.get("encoder_r_bx", true, true)),
  Bx2_(new mblas::Tensor( Bx1_->dim(0), Bx1_->dim(1), Bx1_->dim(2), Bx1_->dim(3), true)),
  Ux_(model.get("encoder_r_Ux", true)),
  Gamma_1_(model.get("encoder_r_gamma1", false)),
  Gamma_2_(model.get("encoder_r_gamma2", false))
{}

std::string Weights::EncBackwardGRU::Debug(unsigned verbosity) const
{
  stringstream strm;
  strm << "W_=" << W_->Debug(verbosity) << endl;
  strm << "B_=" << B_->Debug(verbosity) << endl;
  strm << "U_=" << U_->Debug(verbosity) << endl;
  strm << "Gamma_1_=" << Gamma_1_->Debug(verbosity) << endl;
  strm << "Gamma_2_" << Gamma_2_->Debug(verbosity) << endl;

  return strm.str();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
Weights::DecEmbeddings::DecEmbeddings(const NpzConverter& model)
: E_(model.getFirstOfMany({std::make_pair("Wemb_dec", false),
                           std::make_pair("Wemb", false)}, true))
{}

std::string Weights::DecEmbeddings::Debug(unsigned verbosity) const
{
  stringstream strm;
  strm << "E_=" << E_->Debug(verbosity) << endl;

  return strm.str();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
Weights::DecInit::DecInit(const NpzConverter& model)
: Wi_(model.get("ff_state_W", true)),
  Bi_(model.get("ff_state_b", true, true)),
  Gamma_(model.get("ff_state_gamma", false))
{}

std::string Weights::DecInit::Debug(unsigned verbosity) const
{
  stringstream strm;
  strm << "Wi_=" << Wi_->Debug(verbosity) << endl;
  strm << "Bi_=" << Bi_->Debug(verbosity) << endl;
  strm << "Gamma_=" << Gamma_->Debug(verbosity) << endl;

  return strm.str();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
Weights::DecGRU1::DecGRU1(const NpzConverter& model)
: W_(model.get("decoder_W", true)),
  B_(model.get("decoder_b", true, true)),
  U_(model.get("decoder_U", true)),
  Wx_(model.get("decoder_Wx", true)),
  Bx1_(model.get("decoder_bx", true, true)),
  Bx2_(new mblas::Tensor(Bx1_->dim(0), Bx1_->dim(1), Bx1_->dim(2), Bx1_->dim(3), true)),
  Ux_(model.get("decoder_Ux", true)),
  Gamma_1_(model.get("decoder_cell1_gamma1", false)),
  Gamma_2_(model.get("decoder_cell1_gamma2", false))
{}

std::string Weights::DecGRU1::Debug(unsigned verbosity) const
{
  stringstream strm;
  strm << "W_=" << W_->Debug(verbosity) << endl;
  strm << "B_=" << B_->Debug(verbosity) << endl;
  strm << "U_=" << U_->Debug(verbosity) << endl;
  strm << "Wx_=" << Wx_->Debug(verbosity) << endl;
  strm << "Bx1_=" << Bx1_->Debug(verbosity) << endl;
  strm << "Bx2_=" << Bx2_->Debug(verbosity) << endl;
  strm << "Ux_=" << Ux_->Debug(verbosity) << endl;
  strm << "Gamma_1_=" << Gamma_1_->Debug(verbosity) << endl;
  strm << "Gamma_2_" << Gamma_2_->Debug(verbosity) << endl;

  return strm.str();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
Weights::DecGRU2::DecGRU2(const NpzConverter& model)
: W_(model.get("decoder_Wc", true)),
  B_(model.get("decoder_b_nl", true, true)),
  U_(model.get("decoder_U_nl", true)),
  Wx_(model.get("decoder_Wcx", true)),
  Bx2_(model.get("decoder_bx_nl", true, true)),
  Bx1_(new mblas::Tensor(Bx2_->dim(0), Bx2_->dim(1), Bx2_->dim(2), Bx2_->dim(3), true)),
  Ux_(model.get("decoder_Ux_nl", true)),
  Gamma_1_(model.get("decoder_cell2_gamma1", false)),
  Gamma_2_(model.get("decoder_cell2_gamma2", false))
{}

std::string Weights::DecGRU2::Debug(unsigned verbosity) const
{
  stringstream strm;
  strm << "W_=" << W_->Debug(verbosity) << endl;
  strm << "B_=" << B_->Debug(verbosity) << endl;
  strm << "U_=" << U_->Debug(verbosity) << endl;
  strm << "Wx_=" << Wx_->Debug(verbosity) << endl;
  strm << "Bx1_=" << Bx1_->Debug(verbosity) << endl;
  strm << "Bx2_=" << Bx2_->Debug(verbosity) << endl;
  strm << "Ux_=" << Ux_->Debug(verbosity) << endl;
  strm << "Gamma_1_=" << Gamma_1_->Debug(verbosity) << endl;
  strm << "Gamma_2_" << Gamma_2_->Debug(verbosity) << endl;

  return strm.str();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
Weights::DecLSTM1::DecLSTM1(const NpzConverter& model)
// matrix merging is done to be backwards-compatible with the original LSTM implementation in Amun
// we now use the same format used in Marian
// TODO: adapt to support Nematus LSTM models which use a similar format to Amun's original format
: W_(merge(model.get("decoder_W", true), model.get("decoder_Wx", false))),
  B_(merge(model.get("decoder_b", true, true), model.get("decoder_bx", false, true))),
  U_(merge(model.get("decoder_U", true), model.get("decoder_Ux", false))),
  Gamma_1_(model.get("decoder_cell1_gamma1", false)),
  Gamma_2_(model.get("decoder_cell1_gamma2", false))
{}

std::string Weights::DecLSTM1::Debug(unsigned verbosity) const
{
  stringstream strm;
  strm << "W_=" << W_->Debug(verbosity) << endl;
  strm << "B_=" << B_->Debug(verbosity) << endl;
  strm << "U_=" << U_->Debug(verbosity) << endl;
  strm << "Gamma_1_=" << Gamma_1_->Debug(verbosity) << endl;
  strm << "Gamma_2_" << Gamma_2_->Debug(verbosity) << endl;

  return strm.str();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
Weights::DecLSTM2::DecLSTM2(const NpzConverter& model)
// matrix merging is done to be backwards-compatible with the original LSTM implementation in Amun
// we now use the same format used in Marian
// TODO: adapt to support Nematus LSTM models which use a similar format to Amun's original format
: W_(merge(model.get("decoder_Wc", true), model.get("decoder_Wcx", false))),
  B_(merge(model.get("decoder_b_nl", true, true), model.get("decoder_bx_nl", false, true))),
  U_(merge(model.get("decoder_U_nl", true), model.get("decoder_Ux_nl", false))),
  Gamma_1_(model.get("decoder_cell2_gamma1", false)),
  Gamma_2_(model.get("decoder_cell2_gamma2", false))
{}

std::string Weights::DecLSTM2::Debug(unsigned verbosity) const
{
  stringstream strm;
  strm << "W_=" << W_->Debug(verbosity) << endl;
  strm << "B_=" << B_->Debug(verbosity) << endl;
  strm << "U_=" << U_->Debug(verbosity) << endl;
  strm << "Gamma_1_=" << Gamma_1_->Debug(verbosity) << endl;
  strm << "Gamma_2_" << Gamma_2_->Debug(verbosity) << endl;

  return strm.str();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
Weights::DecAlignment::DecAlignment(const NpzConverter& model)
: V_(model.get("decoder_U_att", true, true)),
  W_(model.get("decoder_W_comb_att", true)),
  B_(model.get("decoder_b_att", true, true)),
  U_(model.get("decoder_Wc_att", true)),
  C_(model.get("decoder_c_tt", true)), // scalar?
  Gamma_1_(model.get("decoder_att_gamma1", false)),
  Gamma_2_(model.get("decoder_att_gamma2", false))
{}

std::string Weights::DecAlignment::Debug(unsigned verbosity) const
{
  stringstream strm;
  strm << "V_=" << V_->Debug(verbosity) << endl;
  strm << "W_=" << W_->Debug(verbosity) << endl;
  strm << "B_=" << B_->Debug(verbosity) << endl;
  strm << "U_=" << U_->Debug(verbosity) << endl;
  strm << "C_=" << C_->Debug(verbosity) << endl;
  strm << "Gamma_1_=" << Gamma_1_->Debug(verbosity) << endl;
  strm << "Gamma_2_" << Gamma_2_->Debug(verbosity) << endl;

  return strm.str();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
Weights::DecSoftmax::DecSoftmax(const NpzConverter& model)
: W1_(model.get("ff_logit_lstm_W", true)),
  B1_(model.get("ff_logit_lstm_b", true, true)),
  W2_(model.get("ff_logit_prev_W", true)),
  B2_(model.get("ff_logit_prev_b", true, true)),
  W3_(model.get("ff_logit_ctx_W", true)),
  B3_(model.get("ff_logit_ctx_b", true, true)),
  W4_(model.getFirstOfMany({std::make_pair(std::string("ff_logit_W"), false),
                            std::make_pair(std::string("Wemb_dec"), true),
                            std::make_pair(std::string("Wemb"), true)}, true)),
  B4_(model.get("ff_logit_b", true, true)),
  Gamma_0_(model.get("ff_logit_l1_gamma0", false)),
  Gamma_1_(model.get("ff_logit_l1_gamma1", false)),
  Gamma_2_(model.get("ff_logit_l1_gamma2", false))
{}

std::string Weights::DecSoftmax::Debug(unsigned verbosity) const
{
  stringstream strm;
  strm << "W1_=" << W1_->Debug(verbosity) << endl;
  strm << "B1_=" << B1_->Debug(verbosity) << endl;
  strm << "W2_=" << W2_->Debug(verbosity) << endl;
  strm << "B2_=" << B2_->Debug(verbosity) << endl;
  strm << "W3_=" << W3_->Debug(verbosity) << endl;
  strm << "B3_=" << B3_->Debug(verbosity) << endl;
  strm << "W4_=" << W4_->Debug(verbosity) << endl;
  strm << "B4_=" << B4_->Debug(verbosity) << endl;

  strm << "Gamma_0_=" << Gamma_0_->Debug(verbosity) << endl;
  strm << "Gamma_1_=" << Gamma_1_->Debug(verbosity) << endl;
  strm << "Gamma_2_" << Gamma_2_->Debug(verbosity) << endl;

  return strm.str();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
Weights::Weights(const std::string& npzFile, const YAML::Node& config,  unsigned device)
: Weights(NpzConverter(npzFile), config, device)
{}

Weights::Weights(const NpzConverter& model, const YAML::Node& config, unsigned device)
: encEmbeddings_(model),
  decEmbeddings_(model),
  decInit_(model),
  decAlignment_(model),
  decSoftmax_(model),
  device_(device)
{

  std::string encCell = config["enc-cell"] ? config["enc-cell"].as<std::string>() : "gru";
  std::string encCell_r = config["enc-cell-r"] ? config["enc-cell-r"].as<std::string>() : encCell;
  initEncForward(model, encCell);
  initEncBackward(model, encCell_r);

  std::string decCell = config["dec-cell"] ? config["dec-cell"].as<std::string>() : "gru";
  std::string decCell2 = config["dec-cell-2"] ? config["dec-cell-2"].as<std::string>() : decCell;
  initDec1(model, decCell);
  initDec2(model, decCell2);
}

void Weights::initEncForward(const NpzConverter& model,std::string celltype){
  if(celltype == "lstm"){
    encForwardLSTM_ = std::shared_ptr<EncForwardLSTM>(new EncForwardLSTM(model));
  } else if (celltype == "mlstm") {
    encForwardMLSTM_ = std::shared_ptr<MultWeights<EncForwardLSTM>>
      (new MultWeights<EncForwardLSTM>(model, "encoder"));
  } else if (celltype == "gru"){
    encForwardGRU_ = std::shared_ptr<EncForwardGRU>(new EncForwardGRU(model));
  }
}

void Weights::initEncBackward(const NpzConverter& model,std::string celltype) {
  if(celltype == "lstm"){
    encBackwardLSTM_ = std::shared_ptr<EncBackwardLSTM>(new EncBackwardLSTM(model));
  } else if (celltype == "mlstm") {
    encBackwardMLSTM_ = std::shared_ptr<MultWeights<EncBackwardLSTM>>
      (new MultWeights<EncBackwardLSTM>(model, "encoder_r"));
  } else if (celltype == "gru"){
    encBackwardGRU_ = std::shared_ptr<EncBackwardGRU>(new EncBackwardGRU(model));
  }
}

void Weights::initDec1(const NpzConverter& model,std::string celltype){
  if (celltype == "lstm"){
    decLSTM1_ = std::shared_ptr<DecLSTM1>(new DecLSTM1(model));
  } else if (celltype == "mlstm") {
    decMLSTM1_ = std::shared_ptr<MultWeights<DecLSTM1>>(new MultWeights<DecLSTM1>(model, "decoder"));
  } else if (celltype == "gru") {
    decGru1_ = std::shared_ptr<DecGRU1>(new DecGRU1(model));
  }
}

void Weights::initDec2(const NpzConverter& model,std::string celltype){
  if (celltype == "lstm"){
    decLSTM2_ = std::shared_ptr<DecLSTM2>(new DecLSTM2(model));
  } else if (celltype == "mlstm") {
    decMLSTM2_ = std::shared_ptr<MultWeights<DecLSTM2>>(new MultWeights<DecLSTM2>(model, "decoder_2"));
  } else if (celltype == "gru") {
    decGru2_ = std::shared_ptr<DecGRU2>(new DecGRU2(model));
  }
}

} // namespace
}

