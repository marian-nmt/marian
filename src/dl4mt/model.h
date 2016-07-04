#pragma once

#include <map>
#include <string>

#include "mblas/matrix.h"
#include "npz_converter.h"

struct Weights {
  
  //////////////////////////////////////////////////////////////////////////////
  
  struct EncEmbeddings {
    EncEmbeddings(const NpzConverter& model);
    
    const mblas::Matrix E_;
  };
  
  struct EncForwardGRU {
    EncForwardGRU(const NpzConverter& model);
    
    const mblas::Matrix W_;
    const mblas::Matrix B_;
    const mblas::Matrix U_;
    const mblas::Matrix Wx_;
    const mblas::Matrix Bx1_;
    const mblas::Matrix Bx2_;
    const mblas::Matrix Ux_;
  };
  
  struct EncBackwardGRU {
    EncBackwardGRU(const NpzConverter& model) ;
    
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
    DecEmbeddings(const NpzConverter& model);
    
    const mblas::Matrix E_;
  };

  struct DecInit {
    DecInit(const NpzConverter& model);
    
    const mblas::Matrix Wi_;
    const mblas::Matrix Bi_;
  };
  
  struct DecGRU1 {
    DecGRU1(const NpzConverter& model);
    
    const mblas::Matrix W_;
    const mblas::Matrix B_;
    const mblas::Matrix U_;
    const mblas::Matrix Wx_;
    const mblas::Matrix Bx1_;
    const mblas::Matrix Bx2_;
    const mblas::Matrix Ux_;
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
  
  const EncEmbeddings encEmbeddings_;
  const DecEmbeddings decEmbeddings_;
  const EncForwardGRU encForwardGRU_;
  const EncBackwardGRU encBackwardGRU_;
  const DecInit decInit_;
  const DecGRU1 decGru1_;
  const DecGRU2 decGru2_;
  const DecAttention decAttention_;
  const DecSoftmax decSoftmax_;
  
  const size_t device_;
};
