#pragma once

#include "mblas/matrix.h"
#include "dl4mt/model.h"
#include "dl4mt/gru.h"
 
class Decoder {
  private:
    template <class Weights>
    class Embeddings {
      public:
        Embeddings(const Weights& model)
        : w_(model)
        {}
            
        void Lookup(mblas::Matrix& rows, const std::vector<size_t>& ids) {
          using namespace mblas;
          std::vector<size_t> tids = ids;
          for(auto&& id : tids)
            if(id >= w_.E_.rows())
              id = 1;
          
          Assemble(rows, w_.E_, tids);
        }
        
        size_t GetCols() {
          return w_.E_.cols();    
        }
        
        size_t GetRows() const {
          return w_.E_.rows();    
        }
        
      private:
        const Weights& w_;
    };
    
    //////////////////////////////////////////////////////////////
    template <class Weights1, class Weights2>
    class RNNHidden {
      public:
        RNNHidden(const Weights1& initModel, const Weights2& gruModel)
        : w_(initModel), gru_(gruModel) {}          
        
        void InitializeState(mblas::Matrix& State,
                             const mblas::Matrix& SourceContext,
                             const size_t batchSize = 1) {
          using namespace mblas;
          namespace bpp = boost::phoenix::placeholders;
          
          Temp_ = SourceContext.colwise().mean().replicate(batchSize, 1);
          State = Temp_ * w_.Wi_;
          BroadcastVec(Tanh(bpp::_1 + bpp::_2), State, w_.Bi_);
        }
        
        void GetNextState(mblas::Matrix& NextState,
                          const mblas::Matrix& State,
                          const mblas::Matrix& Context) {
          gru_.GetNextState(NextState, State, Context);
        }
        
      private:
        const Weights1& w_;
        const GRU<Weights2> gru_;
        
        mblas::Matrix Temp_;
    };
    
    //////////////////////////////////////////////////////////////
    template <class Weights>
    class RNNFinal {
      public:
        RNNFinal(const Weights& model)
        : gru_(model) {}          
        
        void GetNextState(mblas::Matrix& NextState,
                          const mblas::Matrix& State,
                          const mblas::Matrix& Context) {
          gru_.GetNextState(NextState, State, Context);
        }
        
      private:
        const GRU<Weights> gru_;
    };
        
    //////////////////////////////////////////////////////////////
    template <class Weights>
    class Attention {
      public:
        Attention(const Weights& model)
        : w_(model)
        {  }
          
        void GetAlignedSourceContext(mblas::Matrix& AlignedSourceContext,
                                     const mblas::Matrix& HiddenState,
                                     const mblas::Matrix& SourceContext) {
          using namespace mblas;  
          namespace bpp = boost::phoenix::placeholders;
          
          Temp1_ = SourceContext * w_.U_;
          Temp2_ = HiddenState * w_.W_;
          BroadcastVec(bpp::_1 + bpp::_2, Temp2_, w_.B_);
          
          // For batching: create an A across different sentences,
          // maybe by mapping and looping. In the and join different
          // alignment matrices into one
          // Or masking?
          
          Broadcast(Tanh(bpp::_1 + bpp::_2), Temp1_, Temp2_);
          A_ = w_.V_ * Temp1_.transpose();
          size_t words = SourceContext.rows();
          size_t batchSize = HiddenState.rows(); 
          
          A_.resize(words, batchSize);
          A_.transposeInPlace();
          A_.array() += w_.C_(0, 0);
          
          mblas::Softmax(A_);
          
          AlignedSourceContext = A_ * SourceContext;
        }
        
        void GetAttention(mblas::Matrix& Attention) {
          Attention = A_;
        }
      
      private:
        const Weights& w_;
        
        mblas::Matrix Temp1_;
        mblas::Matrix Temp2_;
        mblas::Matrix A_;
        
        mblas::Matrix Ones_;
        mblas::Matrix Sums_;
    };
    
    //////////////////////////////////////////////////////////////
    template <class Weights>
    class Softmax {
      public:
        Softmax(const Weights& model)
        : w_(model), filtered_(false)
        {
          const_cast<mblas::Matrix&>(w_.W1_).transposeInPlace();
          const_cast<mblas::Matrix&>(w_.W2_).transposeInPlace();
          const_cast<mblas::Matrix&>(w_.W3_).transposeInPlace();
          const_cast<mblas::Matrix&>(w_.W4_).transposeInPlace();
        }
          
        void GetProbs(mblas::Matrix& Probs,
                  const mblas::Matrix& State,
                  const mblas::Matrix& Embedding,
                  const mblas::Matrix& AlignedSourceContext) {
          using namespace mblas;
          
          size_t rows = State.rows();
          auto t1 = (w_.W1_ * State.transpose()).colwise() + w_.B1_.transpose();
          auto t2 = (w_.W2_ * Embedding.transpose()).colwise() + w_.B2_.transpose();
          auto t3 = (w_.W3_ * AlignedSourceContext.transpose()).colwise() + w_.B3_.transpose();
          auto t = (t1 + t2 + t3).unaryExpr(&tanhapprox);
          
          //if(!filtered_)
            Probs.noalias() = ((w_.W4_ * t).colwise() + w_.B4_.transpose()).unaryExpr(&expapprox);
            Matrix denoms = Probs.colwise().sum();
            for(size_t i = 0; i < Probs.cols(); ++i)
              Probs.col(i) /= denoms(i);
            Probs = Probs.unaryExpr(&logapprox);
          //else
            //Probs.noalias() = (t *  FilteredW4_).rowwise() + w_.B4_;

          //mblas::SoftmaxLog(Probs);
          //auto nums = Probs.unaryExpr(&expapprox);
          //auto denoms = nums.colwise().sum();
          //
          //for(size_t i = 0; i < nums.cols(); ++i)
          //  Probs.col(i) = (nums.col(i) / denoms(i)).unaryExpr(&logapprox);
        }
    
        void Filter(const std::vector<size_t>& ids) {
          filtered_ = true;
          
          using namespace mblas;
          
          Matrix TempW4 = w_.W4_.transpose();
          Assemble(FilteredW4_, TempW4, ids);
          FilteredW4_.transposeInPlace();
          
          Matrix TempB4 = w_.B4_.transpose();
          Assemble(FilteredB4_, TempB4, ids);
          FilteredB4_.transposeInPlace();
        }
       
      private:        
        const Weights& w_;
        
        bool filtered_;
        mblas::Matrix FilteredW4_;
        mblas::Matrix FilteredB4_;
        
        mblas::Matrix T1_;
        mblas::Matrix T2_;
        mblas::Matrix T3_;
    };
    
  public:
    Decoder(const Weights& model)
    : embeddings_(model.decEmbeddings_),
      rnn1_(model.decInit_, model.decGru1_),
      rnn2_(model.decGru2_),
	  attention_(model.decAttention_),
      softmax_(model.decSoftmax_)
    {}
    
    void MakeStep(mblas::Matrix& NextState,
                  mblas::Matrix& Probs,
                  const mblas::Matrix& State,
                  const mblas::Matrix& Embeddings,
                  const mblas::Matrix& SourceContext) {
      GetHiddenState(HiddenState_, State, Embeddings);
      GetAlignedSourceContext(AlignedSourceContext_, HiddenState_, SourceContext);
      GetNextState(NextState, HiddenState_, AlignedSourceContext_);
      GetProbs(Probs, NextState, Embeddings, AlignedSourceContext_);
    }
    
    void EmptyState(mblas::Matrix& State,
                    const mblas::Matrix& SourceContext,
                    size_t batchSize = 1) {
      rnn1_.InitializeState(State, SourceContext, batchSize);
    }
    
    void EmptyEmbedding(mblas::Matrix& Embedding,
                        size_t batchSize = 1) {
      Embedding.resize(batchSize, embeddings_.GetCols());
      Embedding.fill(0);
    }
    
    void Lookup(mblas::Matrix& Embedding,
                const std::vector<size_t>& w) {
      embeddings_.Lookup(Embedding, w);
    }
    
    void Filter(const std::vector<size_t>& ids) {
      softmax_.Filter(ids);
    }
      
    void GetAttention(mblas::Matrix& attention) {
    	attention_.GetAttention(attention);
    }
    
    size_t GetVocabSize() const {
      return embeddings_.GetRows();
    }
    
  private:
    
    void GetHiddenState(mblas::Matrix& HiddenState,
                        const mblas::Matrix& PrevState,
                        const mblas::Matrix& Embedding) {
      rnn1_.GetNextState(HiddenState, PrevState, Embedding);
    }
    
    void GetAlignedSourceContext(mblas::Matrix& AlignedSourceContext,
                                 const mblas::Matrix& HiddenState,
                                 const mblas::Matrix& SourceContext) {
    	attention_.GetAlignedSourceContext(AlignedSourceContext, HiddenState, SourceContext);
    }
    
    void GetNextState(mblas::Matrix& State,
                      const mblas::Matrix& HiddenState,
                      const mblas::Matrix& AlignedSourceContext) {
      rnn2_.GetNextState(State, HiddenState, AlignedSourceContext);
    }
    
    
    void GetProbs(mblas::Matrix& Probs,
                  const mblas::Matrix& State,
                  const mblas::Matrix& Embedding,
                  const mblas::Matrix& AlignedSourceContext) {
      softmax_.GetProbs(Probs, State, Embedding, AlignedSourceContext);
    }
    
  private:
    mblas::Matrix HiddenState_;
    mblas::Matrix AlignedSourceContext_;  
    
    Embeddings<Weights::Embeddings> embeddings_;
    RNNHidden<Weights::DecInit, Weights::GRU> rnn1_;
    RNNFinal<Weights::DecGRU2> rnn2_;
    Attention<Weights::DecAttention> attention_;
    Softmax<Weights::DecSoftmax> softmax_;
};
