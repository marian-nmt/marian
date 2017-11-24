#pragma once

#include "../mblas/matrix.h"
#include "../dl4mt/model.h"
#include "../dl4mt/gru.h"

namespace amunmt {
namespace CPU {
namespace dl4mt {

class Encoder {
  private:

	/////////////////////////////////////////////////////////////////
    template <class Weights>
    class Embeddings {
      public:
        Embeddings(const Weights& model)
        : w_(model)
        {}
          
        void Lookup(mblas::Matrix& Row, size_t i) {
		  size_t len = w_.E_.columns();
          if(i < w_.E_.rows())
            Row = blaze::submatrix(w_.E_, i, 0, 1, len);
          else
            Row = blaze::submatrix(w_.E_, 1, 0, 1, len); // UNK
        }
      
        const Weights& w_;
      private:
    };
    
    /////////////////////////////////////////////////////////////////
    template <class Weights>
    class RNN {
      public:
        RNN(const Weights& model)
        : gru_(model) {}
        
        void InitializeState(size_t batchSize = 1) {
          State_.resize(batchSize, gru_.GetStateLength());
		  State_ = 0.0f;
        }
        
        void GetNextState(mblas::Matrix& NextState,
                          const mblas::Matrix& State,
                          const mblas::Matrix& Embd) {
          gru_.GetNextState(NextState, State, Embd);
        }
        
        template <class It>
        void Encode(It it, It end,
                        mblas::Matrix& Context, bool invert) {
          InitializeState();
          
          size_t n = std::distance(it, end);
          size_t i = 0;
          while(it != end) {
            GetNextState(State_, State_, *it++);
            
			size_t len = gru_.GetStateLength();
            if(invert)
              blaze::submatrix(Context, n - i - 1, len, 1, len) = State_;
            else
			  blaze::submatrix(Context, i, 0, 1, len) = State_;
            ++i;
          }
        }
        
        size_t GetStateLength() const {
          return gru_.GetStateLength();
        }
        
      private:
        // Model matrices
        const GRU<Weights> gru_;
        
        mblas::Matrix State_;
    };
    
  /////////////////////////////////////////////////////////////////
  public:
    Encoder(const Weights& model)
    : embeddings_(model.encEmbeddings_),
      forwardRnn_(model.encForwardGRU_),
      backwardRnn_(model.encBackwardGRU_)
    {}
    
    void Encode(const std::vector<uint>& words,
                    mblas::Matrix& context);
    
  private:
    Embeddings<Weights::Embeddings> embeddings_;
    RNN<Weights::GRU> forwardRnn_;
    RNN<Weights::GRU> backwardRnn_;
};

}
}
}

