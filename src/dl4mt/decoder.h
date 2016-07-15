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
          auto mean = SourceContext.colwise().mean();
          auto init = mean.colwise().replicate(batchSize);
          State = (init * w_.Wi_).rowwise() + w_.Bi_;
        }
        
        void GetNextState(mblas::Matrix& NextState,
                          const mblas::Matrix& State,
                          const mblas::Matrix& Context) {
          gru_.GetNextState(NextState, State, Context);
        }
        
      private:
        const Weights1& w_;
        const GRU<Weights2> gru_;
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
          
          Temp1_.noalias() = SourceContext * w_.U_;
          Temp2_.noalias() = (HiddenState * w_.W_).rowwise() + w_.B_;
          
          Broadcast(Tanh(bpp::_1 + bpp::_2), Temp1_, Temp2_);
          A_.noalias() = w_.V_ * Temp1_.transpose();
          
          size_t words = SourceContext.rows();
          size_t batchSize = HiddenState.rows(); 
          A_.resize(words, batchSize);
          A_.array() += w_.C_(0, 0);
          A_.transposeInPlace();
          
          Matrix nums = A_.unaryExpr(&expapprox);
          Matrix denoms = nums.rowwise().sum();
          for(size_t i = 0; i < nums.rows(); ++i)
            A_.row(i) = nums.row(i) / denoms(i);
          
          AlignedSourceContext.noalias() = A_ * SourceContext;
        }
        
        void GetAttention(mblas::Matrix& Attention) {
          Attention = A_;
        }
      
      private:
        const Weights& w_;
        
        mblas::Matrix Temp1_;
        mblas::Matrix Temp2_;
        mblas::Matrix A_;
    };
    
    //////////////////////////////////////////////////////////////
    template <class Weights>
    class Softmax {
      public:
        Softmax(const Weights& model)
        : w_(model), filtered_(false)
        {
          const_cast<mblas::Matrix&>(w_.W4_).transposeInPlace();
          RB4_ = w_.B4_.transpose();
        }
          
        void GetProbs(mblas::Matrix& Probs,
                  const mblas::Matrix& State,
                  const mblas::Matrix& Embedding,
                  const mblas::Matrix& AlignedSourceContext) {          

          mblas::Matrix t1 = State * w_.W1_;
          T1_.noalias() = t1.rowwise() + w_.B1_;
          T2_.noalias() = (Embedding * w_.W2_).rowwise() + w_.B2_;
          T3_.noalias() = (AlignedSourceContext * w_.W3_).rowwise() + w_.B3_;
          mblas::Matrix t = (T1_ + T2_ + T3_).unaryExpr(&tanhapprox).transpose();
          
          if(!filtered_)
            Probs.noalias() = ((w_.W4_ * t).colwise() + RB4_).unaryExpr(&expapprox);
          else
            Probs.noalias() = ((FilteredW4_ * t).colwise() + FilteredB4_).unaryExpr(&expapprox);

          mblas::Matrix denoms = Probs.colwise().sum();
          for(size_t i = 0; i < Probs.cols(); ++i)
            Probs.col(i) /= denoms(i);
          Probs = Probs.unaryExpr(&logapprox);
        }
    
        void Filter(const std::vector<size_t>& ids) {
          filtered_ = true;
          using namespace mblas;
          Assemble(FilteredW4_, w_.W4_, ids);
          Assemble(FilteredB4_, RB4_, ids);
        }
       
      private:        
        const Weights& w_;
        mblas::Matrix T1_;
        mblas::Matrix T2_;
        mblas::Matrix T3_;
        
        bool filtered_;
        mblas::RVector RB4_;
        mblas::Matrix FilteredW4_;
        mblas::RVector FilteredB4_;
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
