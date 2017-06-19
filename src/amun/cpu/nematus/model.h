#pragma once

#include <iostream>
#include <map>
#include <string>

#include "../npz_converter.h"

#include "../mblas/matrix.h"

namespace amunmt {
namespace CPU {

struct Weights {
  class Transition {
    public:
      enum class TransitionType {Encoder, Decoder};
      Transition(const NpzConverter& model, TransitionType type, std::string prefix,
                 std::string infix="")
        : depth_(findTransitionDepth(model, prefix, infix)), type_(type)
      {
        for (int i = 1; i <= depth_; ++i) {
          U_.emplace_back(model[name(prefix, "U", infix, i)]);
          Ux_.emplace_back(model[name(prefix, "Ux", infix, i)]);
          B_.emplace_back(model(name(prefix, "b", infix, i), true));
          U_lns_.emplace_back(model[name(prefix, "U", infix, i, "_lns")]);
          U_lnb_.emplace_back(model[name(prefix, "U", infix, i, "_lnb")]);
          Ux_lns_.emplace_back(model[name(prefix, "Ux", infix, i, "_lns")]);
          Ux_lnb_.emplace_back(model[name(prefix, "Ux", infix, i, "_lnb")]);
          // decoder_U_nl_drt_4_lnb
          switch(type) {
            case TransitionType::Encoder:
              Bx1_.emplace_back(1, Ux_.back().Cols());
              const_cast<mblas::Matrix&>(Bx1_.back()) = 0.0f;
              Bx2_.emplace_back(model(name(prefix, "bx", infix, i), true));
              break;
            case TransitionType::Decoder:
              Bx1_.emplace_back(model(name(prefix, "bx", infix, i), true));
              Bx2_.emplace_back(1, Ux_.back().Cols());
              const_cast<mblas::Matrix&>(Bx2_.back()) = 0.0f;
              break;
          }
        }
      }

    static int findTransitionDepth(const NpzConverter& model, std::string prefix, std::string infix) {
      int currentDepth = 0;
      while (true) {
        if (model.has(prefix + "b" + infix + "_drt_" + std::to_string(currentDepth + 1))) {
          ++currentDepth;
        } else {
          break;
        }
      }
      std::cerr << "Found transition depth: " << currentDepth << std::endl;
      return currentDepth;
    }

    int size() const {
      return depth_;
    }

    TransitionType type() const {
      return type_;
    }

    protected:
      std::string name(const std::string& prefix, std::string name, std::string infix, int index,
          std::string suffix = "")
      {
        return prefix + name + infix + "_drt_" + std::to_string(index) + suffix;
      }

    private:
      int depth_;
      TransitionType type_;

    public:
      std::vector<mblas::Matrix> B_;
      std::vector<mblas::Matrix> Bx1_;
      std::vector<mblas::Matrix> Bx2_;
      std::vector<mblas::Matrix> U_;
      std::vector<mblas::Matrix> Ux_;

      std::vector<mblas::Matrix> U_lns_;
      std::vector<mblas::Matrix> U_lnb_;
      std::vector<mblas::Matrix> Ux_lns_;
      std::vector<mblas::Matrix> Ux_lnb_;

  };

  struct Embeddings {
    Embeddings(const NpzConverter& model, const std::string &key);
    Embeddings(const NpzConverter& model, const std::vector<std::pair<std::string, bool>> keys);

    const mblas::Matrix E_;
  };

  struct GRU {
    GRU(const NpzConverter& model, std::string prefix, std::vector<std::string> keys);

    const mblas::Matrix W_;
    const mblas::Matrix B_;
    const mblas::Matrix U_;
    const mblas::Matrix Wx_;
    const mblas::Matrix Bx1_;
    const mblas::Matrix Bx2_;
    const mblas::Matrix Bx3_;
    const mblas::Matrix Ux_;

    const mblas::Matrix W_lns_;
    const mblas::Matrix W_lnb_;
    const mblas::Matrix Wx_lns_;
    const mblas::Matrix Wx_lnb_;
    const mblas::Matrix U_lns_;
    const mblas::Matrix U_lnb_;
    const mblas::Matrix Ux_lns_;
    const mblas::Matrix Ux_lnb_;
  };

  struct DecInit {
    DecInit(const NpzConverter& model);

    const mblas::Matrix Wi_;
    const mblas::Matrix Bi_;
    const mblas::Matrix lns_;
    const mblas::Matrix lnb_;
  };

  struct DecGRU2 {
    DecGRU2(const NpzConverter& model, std::string prefix, std::vector<std::string> keys);

    const mblas::Matrix W_;
    const mblas::Matrix B_;
    const mblas::Matrix U_;
    const mblas::Matrix Wx_;
    const mblas::Matrix Bx3_;
    const mblas::Matrix Bx2_;
    const mblas::Matrix Bx1_;
    const mblas::Matrix Ux_;

    const mblas::Matrix W_lns_;
    const mblas::Matrix W_lnb_;
    const mblas::Matrix Wx_lns_;
    const mblas::Matrix Wx_lnb_;
    const mblas::Matrix U_lns_;
    const mblas::Matrix U_lnb_;
    const mblas::Matrix Ux_lns_;
    const mblas::Matrix Ux_lnb_;
  };

  struct DecAttention {
    DecAttention(const NpzConverter& model);

    const mblas::Matrix V_;
    const mblas::Matrix W_;
    const mblas::Matrix B_;
    const mblas::Matrix U_;
    const mblas::Matrix C_;
    const mblas::Matrix Wc_att_lns_;
    const mblas::Matrix Wc_att_lnb_;
    const mblas::Matrix W_comb_lns_;
    const mblas::Matrix W_comb_lnb_;
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
    const mblas::Matrix lns_1_;
    const mblas::Matrix lns_2_;
    const mblas::Matrix lns_3_;
    const mblas::Matrix lnb_1_;
    const mblas::Matrix lnb_2_;
    const mblas::Matrix lnb_3_;
  };


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
  const Transition encForwardTransition_;
  const Transition encBackwardTransition_;
  const Transition decTransition_;
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

