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
    EncEmbeddings(const NpzConverter& model);

    const std::shared_ptr<mblas::Matrix> E_;
  };

  ////////////////////////////////////////////////////////////////////////////////////////////////////
  struct EncForwardGRU {
    EncForwardGRU(const EncForwardGRU&) = delete;
    EncForwardGRU(const NpzConverter& model);

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

  ////////////////////////////////////////////////////////////////////////////////////////////////////
  struct EncBackwardGRU {
    EncBackwardGRU(const EncBackwardGRU&) = delete;
    EncBackwardGRU(const NpzConverter& model);

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

  ////////////////////////////////////////////////////////////////////////////////////////////////////
  struct EncForwardLSTM {
    EncForwardLSTM(const EncForwardLSTM&) = delete;
    EncForwardLSTM(const NpzConverter& model);

    const std::shared_ptr<mblas::Matrix> W_;
    const std::shared_ptr<mblas::Matrix> B_;
    const std::shared_ptr<mblas::Matrix> U_;
    const std::shared_ptr<mblas::Matrix> Wx_;
    const std::shared_ptr<mblas::Matrix> Bx_;
    const std::shared_ptr<mblas::Matrix> Ux_;
    const std::shared_ptr<mblas::Matrix> Gamma_1_;
    const std::shared_ptr<mblas::Matrix> Gamma_2_;
  };

  ////////////////////////////////////////////////////////////////////////////////////////////////////
  struct EncBackwardLSTM {
    EncBackwardLSTM(const EncBackwardLSTM&) = delete;
    EncBackwardLSTM(const NpzConverter& model);

    const std::shared_ptr<mblas::Matrix> W_;
    const std::shared_ptr<mblas::Matrix> B_;
    const std::shared_ptr<mblas::Matrix> U_;
    const std::shared_ptr<mblas::Matrix> Wx_;
    const std::shared_ptr<mblas::Matrix> Bx_;
    const std::shared_ptr<mblas::Matrix> Ux_;
    const std::shared_ptr<mblas::Matrix> Gamma_1_;
    const std::shared_ptr<mblas::Matrix> Gamma_2_;
  };

  ////////////////////////////////////////////////////////////////////////////////////////////////////

  struct DecEmbeddings {
    DecEmbeddings(const DecEmbeddings&) = delete;
    DecEmbeddings(const NpzConverter& model);

    const std::shared_ptr<mblas::Matrix> E_;
  };

  ////////////////////////////////////////////////////////////////////////////////////////////////////

  struct DecInit {
    DecInit(const DecInit&) = delete;
    DecInit(const NpzConverter& model);

    const std::shared_ptr<mblas::Matrix> Wi_;
    const std::shared_ptr<mblas::Matrix> Bi_;
    const std::shared_ptr<mblas::Matrix> Gamma_;
  };

  ////////////////////////////////////////////////////////////////////////////////////////////////////

  struct DecGRU1 {
    DecGRU1(const DecGRU1&) = delete;
    DecGRU1(const NpzConverter& model);

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

  ////////////////////////////////////////////////////////////////////////////////////////////////////

  struct DecGRU2 {
    DecGRU2(const DecGRU2&) = delete;
    DecGRU2(const NpzConverter& model);

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

  ////////////////////////////////////////////////////////////////////////////////////////////////////

  struct DecLSTM1 {
    DecLSTM1(const DecLSTM1&) = delete;
    DecLSTM1(const NpzConverter& model);

    const std::shared_ptr<mblas::Matrix> W_;
    const std::shared_ptr<mblas::Matrix> B_;
    const std::shared_ptr<mblas::Matrix> U_;
    const std::shared_ptr<mblas::Matrix> Wx_;
    const std::shared_ptr<mblas::Matrix> Bx_;
    const std::shared_ptr<mblas::Matrix> Ux_;
    const std::shared_ptr<mblas::Matrix> Gamma_1_;
    const std::shared_ptr<mblas::Matrix> Gamma_2_;
  };

  ////////////////////////////////////////////////////////////////////////////////////////////////////

  struct DecLSTM2 {
    DecLSTM2(const DecLSTM2&) = delete;
    DecLSTM2(const NpzConverter& model);

    const std::shared_ptr<mblas::Matrix> W_;
    const std::shared_ptr<mblas::Matrix> B_;
    const std::shared_ptr<mblas::Matrix> U_;
    const std::shared_ptr<mblas::Matrix> Wx_;
    const std::shared_ptr<mblas::Matrix> Bx_;
    const std::shared_ptr<mblas::Matrix> Ux_;
    const std::shared_ptr<mblas::Matrix> Gamma_1_;
    const std::shared_ptr<mblas::Matrix> Gamma_2_;
  };

  ////////////////////////////////////////////////////////////////////////////////////////////////////

  // A wrapper class to deserialize weights for multiplicative-LSTM,
  // multiplicative-GRU and such
  template<class BaseWeights>
  struct MultWeights: public BaseWeights {
    MultWeights(const MultWeights&) = delete;
    MultWeights(const NpzConverter& model, const std::string& prefix)
      : BaseWeights(model),
      Wm_(model.get(p(prefix, "Wm"), true)),
      Bm_(model.get(p(prefix, "bm"), true, true)),
      Um_(model.get(p(prefix, "Um"), true)),
      Bmu_(model.get(p(prefix, "bmu"), true, true))
      {}
    const std::shared_ptr<mblas::Matrix> Wm_;
    const std::shared_ptr<mblas::Matrix> Bm_;
    const std::shared_ptr<mblas::Matrix> Um_;
    const std::shared_ptr<mblas::Matrix> Bmu_;
  private:
    std::string p(std::string prefix, std::string sufix){
      return prefix + "_" + sufix;
    }
  };

  ////////////////////////////////////////////////////////////////////////////////////////////////////

  struct DecAlignment {
    DecAlignment(const DecAlignment&) = delete;
    DecAlignment(const NpzConverter& model);

    const std::shared_ptr<mblas::Matrix> V_;
    const std::shared_ptr<mblas::Matrix> W_;
    const std::shared_ptr<mblas::Matrix> B_;
    const std::shared_ptr<mblas::Matrix> U_;
    const std::shared_ptr<mblas::Matrix> C_;
    const std::shared_ptr<mblas::Matrix> Gamma_1_;
    const std::shared_ptr<mblas::Matrix> Gamma_2_;
  };

  ////////////////////////////////////////////////////////////////////////////////////////////////////

  struct DecSoftmax {
    DecSoftmax(const DecSoftmax&) = delete;
    DecSoftmax(const NpzConverter& model);

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

  ////////////////////////////////////////////////////////////////////////////////////////////////////

  Weights(const std::string& npzFile, const YAML::Node& config,  size_t device);

  Weights(const NpzConverter& model, const YAML::Node& config, size_t device);

  Weights(const Weights&) = delete;

  size_t GetDevice() {
    return device_;
  }

private:
  void initEncForward(const NpzConverter& model,std::string celltype);
  void initEncBackward(const NpzConverter& model,std::string celltype);
  void initDec1(const NpzConverter& model,std::string celltype);
  void initDec2(const NpzConverter& model,std::string celltype);

public:
  const EncEmbeddings encEmbeddings_;
  const DecEmbeddings decEmbeddings_;
  // of these usuall only two at a time will be not null
  std::shared_ptr<EncForwardGRU> encForwardGRU_;
  std::shared_ptr<EncBackwardGRU> encBackwardGRU_;
  std::shared_ptr<EncForwardLSTM> encForwardLSTM_;
  std::shared_ptr<EncBackwardLSTM> encBackwardLSTM_;
  std::shared_ptr<MultWeights<EncForwardLSTM>> encForwardMLSTM_;
  std::shared_ptr<MultWeights<EncBackwardLSTM>> encBackwardMLSTM_;
  const DecInit decInit_;
  // of these usuall only two at a time will be not null
  std::shared_ptr<DecGRU1> decGru1_;
  std::shared_ptr<DecGRU2> decGru2_;
  std::shared_ptr<DecLSTM1> decLSTM1_;
  std::shared_ptr<DecLSTM2> decLSTM2_;
  std::shared_ptr<MultWeights<DecLSTM1>> decMLSTM1_;
  std::shared_ptr<MultWeights<DecLSTM2>> decMLSTM2_;
  const DecAlignment decAlignment_;
  const DecSoftmax decSoftmax_;

  const size_t device_;
};

}
}


