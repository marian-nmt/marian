#pragma once

#include "mblas/matrix.h"
#include "dl4mt/model.h"
#include "dl4mt/gru.h"
 
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
          if(i < w_.E_.rows())
            Row = w_.E_.row(i);
          else
            Row = w_.E_.row(1);
        }
      
      private:
        const Weights& w_;
    };
    
    /////////////////////////////////////////////////////////////////
    template <class Weights>
    class RNN {
      public:
        RNN(const Weights& model)
        : gru_(model) {}
        
        void InitializeState(size_t batchSize = 1) {
          State_ = Eigen::ArrayXXf::Zero(batchSize, gru_.GetStateLength());
        }
        
        void GetNextState(mblas::Matrix& NextState,
                          const mblas::Matrix& State,
                          const mblas::Matrix& Embd) {
          gru_.GetNextState(NextState, State, Embd);
        }
        
        template <class It>
        void GetContext(It it, It end, 
                        mblas::Matrix& Context, bool invert) {
          InitializeState();
          
          size_t n = std::distance(it, end);
          size_t i = 0;
          while(it != end) {
            GetNextState(State_, State_, *it++);
            
            if(invert)
			  Context.block(n - i - 1, gru_.GetStateLength(),
							1, gru_.GetStateLength()) = State_;
            else
			  Context.block(i, 0,
							1, gru_.GetStateLength()) = State_;
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
    
    void GetContext(const std::vector<size_t>& words,
                    mblas::Matrix& context);
    
  private:
    Embeddings<Weights::Embeddings> embeddings_;
    RNN<Weights::GRU> forwardRnn_;
    RNN<Weights::GRU> backwardRnn_;
};
