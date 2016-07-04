#pragma once

#include <iostream>
#include <map>
#include <string>

#include "mblas/matrix.h"
#include "npz_converter.h"

struct Weights {
  
  //////////////////////////////////////////////////////////////////////////////
  
  struct Embeddings {
    Embeddings(const NpzConverter& model, const std::string &key);
    
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
  };
    
  //////////////////////////////////////////////////////////////////////////////
  
  struct DecInit {
    DecInit(const NpzConverter& model);
    
    const mblas::Matrix Wi_;
    const mblas::Matrix Bi_;
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
  };
  
  struct DecAttention {
    DecAttention(const NpzConverter& model);
          
    const mblas::Matrix V_;
    const mblas::Matrix W_;
    const mblas::Matrix B_;
    const mblas::Matrix U_;
    const mblas::Matrix C_;
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
  };
  
  //////////////////////////////////////////////////////////////////////////////

  Weights(const std::string& npzFile, size_t device = 0)
  : Weights(NpzConverter(npzFile), device)
  {}
  
  Weights(const NpzConverter& model, size_t device = 0);
  
  size_t GetDevice() {
    return device_;
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
  
  const size_t device_;
};

inline std::ostream& operator<<(std::ostream &out, const Weights::Embeddings &obj)
{
	out << "E_ = " << obj.E_;
	return out;
}

inline std::ostream& operator<<(std::ostream &out, const Weights::GRU &obj)
{
	out << "W_ =" << obj.W_ << std::endl;
	out << "B_ =" << obj.B_ << std::endl;
	out << "U_ =" << obj.U_ << std::endl;
	out << "Wx_ =" << obj.Wx_ << std::endl;
	out << "Bx1_ =" << obj.Bx1_ << std::endl;
	out << "Bx2_ =" << obj.Bx2_ << std::endl;
	out << "Ux_ =" << obj.Ux_;
	return out;
}

inline std::ostream& operator<<(std::ostream &out, const Weights &obj)
{
	out << "encEmbeddings_ =" << obj.encEmbeddings_ << std::endl;
	out << "decEmbeddings_ =" << obj.decEmbeddings_ << std::endl;

	out << "encForwardGRU_ =" << obj.encForwardGRU_ << std::endl;
	out << "encBackwardGRU_ =" << obj.encBackwardGRU_ << std::endl;

	return out;
}

