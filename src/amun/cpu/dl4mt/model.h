#pragma once

#include <iostream>
#include <map>
#include <string>

#include "cpu/npz_converter.h"
#include "cpu/mblas/matrix.h"

namespace amunmt {
namespace CPU {
namespace dl4mt {

struct Weights {

  //////////////////////////////////////////////////////////////////////////////

  struct Embeddings {
    Embeddings(const NpzConverter& model, const std::string &key);
    Embeddings(const NpzConverter& model, const std::vector<std::pair<std::string, bool>> keys);

    const mblas::Matrix E_;
  };

  struct GRU {
	GRU(const NpzConverter& model, const std::vector<std::string> &keys);

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

  struct DecInit {
    DecInit(const NpzConverter& model);

    const mblas::Matrix Wi_;
    const mblas::Matrix Bi_;
    const mblas::Matrix Gamma_;
  };

  struct DecGRU2 {
    DecGRU2(const NpzConverter& model);

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

  struct DecAttention {
    DecAttention(const NpzConverter& model);

    const mblas::Matrix V_;
    const mblas::Matrix W_;
    const mblas::Matrix B_;
    const mblas::Matrix U_;
    const mblas::Matrix C_;
    const mblas::Matrix Gamma_1_;
    const mblas::Matrix Gamma_2_;
  };

  struct DecSoftmax {
    DecSoftmax(const NpzConverter& model);

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

  Weights(const std::string& npzFile, size_t device = 0)
    : Weights(NpzConverter(npzFile), device)
  {}

  Weights(const NpzConverter& model, size_t device = 0);

  size_t GetDevice() {
    return std::numeric_limits<size_t>::max();
  }

  const Embeddings encEmbeddings_;
  const Embeddings decEmbeddings_;
  const GRU encForwardGRU_;
  const GRU encBackwardGRU_;
  const DecInit decInit_;
  const GRU decGru1_;
  const DecGRU2 decGru2_;
  const DecAttention decAttention_;
  const DecSoftmax decSoftmax_;
};

inline std::ostream& operator<<(std::ostream &out, const Weights::Embeddings &obj)
{
	out << "E_ \t" << obj.E_;
	return out;
}

inline std::ostream& operator<<(std::ostream &out, const Weights::GRU &obj)
{
	out << "W_ \t" << obj.W_ << std::endl;
	out << "B_ \t" << obj.B_ << std::endl;
	out << "U_ \t" << obj.U_ << std::endl;
	out << "Wx_ \t" << obj.Wx_ << std::endl;
	out << "Bx1_ \t" << obj.Bx1_ << std::endl;
	out << "Bx2_ \t" << obj.Bx2_ << std::endl;
	out << "Ux_ \t" << obj.Ux_;
	return out;
}

inline std::ostream& operator<<(std::ostream &out, const Weights::DecGRU2 &obj)
{
	out << "W_ \t" << obj.W_ << std::endl;
	out << "B_ \t" << obj.B_ << std::endl;
	out << "U_ \t" << obj.U_ << std::endl;
	out << "Wx_ \t" << obj.Wx_ << std::endl;
	out << "Bx1_ \t" << obj.Bx1_ << std::endl;
	out << "Bx2_ \t" << obj.Bx2_ << std::endl;
	out << "Ux_ \t" << obj.Ux_;
	return out;
}

inline std::ostream& operator<<(std::ostream &out, const Weights::DecInit &obj)
{
	out << "Wi_ \t" << obj.Wi_ << std::endl;
	out << "Bi_ \t" << obj.Bi_ ;
	return out;
}

inline std::ostream& operator<<(std::ostream &out, const Weights::DecAttention &obj)
{
	out << "V_ \t" << obj.V_ << std::endl;
	out << "W_ \t" << obj.W_ << std::endl;
	out << "B_ \t" << obj.B_ << std::endl;
	out << "U_ \t" << obj.U_ << std::endl;
	out << "C_ \t" << obj.C_ ;
	return out;
}

inline std::ostream& operator<<(std::ostream &out, const Weights::DecSoftmax &obj)
{
	out << "W1_ \t" << obj.W1_ << std::endl;
	out << "B1_ \t" << obj.B1_ << std::endl;
	out << "W2_ \t" << obj.W2_ << std::endl;
	out << "B2_ \t" << obj.B2_ << std::endl;
	out << "W3_ \t" << obj.W3_ << std::endl;
	out << "B3_ \t" << obj.B3_ << std::endl;
	out << "W4_ \t" << obj.W4_ << std::endl;
	out << "B4_ \t" << obj.B4_ ;

	return out;
}

inline std::ostream& operator<<(std::ostream &out, const Weights &obj)
{
	out << "\n encEmbeddings_ \n" << obj.encEmbeddings_ << std::endl;
	out << "\n decEmbeddings_ \n" << obj.decEmbeddings_ << std::endl;

	out << "\n encForwardGRU_ \n" << obj.encForwardGRU_ << std::endl;
	out << "\n encBackwardGRU_ \n" << obj.encBackwardGRU_ << std::endl;

	out << "\n decInit_ \n" << obj.decInit_ << std::endl;

	out << "\n decGru1_ \n" << obj.decGru1_ << std::endl;
	out << "\n decGru2_ \n" << obj.decGru2_ << std::endl;

	out << "\n decAttention_ \n" << obj.decAttention_ << std::endl;

	out << "\n decSoftmax_ \n" << obj.decSoftmax_ << std::endl;

	//Debug2(obj.encEmbeddings_.E_);

	return out;
}

}
}
}

