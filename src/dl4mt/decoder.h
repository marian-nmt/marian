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
            
        void Lookup(mblas::Matrix& Rows, const std::vector<size_t>& ids) {
          using namespace mblas;
          std::vector<size_t> tids = ids;
          for(auto&& id : tids)
            if(id >= w_.E_.Rows())
              id = 1;
          Assemble(Rows, w_.E_, tids);
        }
        
        size_t GetCols() {
          return w_.E_.Cols();    
        }
        
        size_t GetRows() const {
          return w_.E_.Rows();    
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
          
          // calculate mean of source context, rowwise
          Mean(Temp1_, SourceContext);
          
          // Repeat mean batchSize times by broadcasting
          Temp2_.Clear();
          Temp2_.Resize(batchSize, SourceContext.Cols(), 0.0);
          BroadcastVec(bpp::_1 + bpp::_2, Temp2_, Temp1_);
          
          Prod(State, Temp2_, w_.Wi_);
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
        
        mblas::Matrix Temp1_;
        mblas::Matrix Temp2_;
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
          
          Prod(Temp1_, SourceContext, w_.U_);
          Prod(Temp2_, HiddenState, w_.W_);
          BroadcastVec(bpp::_1 + bpp::_2, Temp2_, w_.B_);
          
          // For batching: create an A across different sentences,
          // maybe by mapping and looping. In the and join different
          // alignment matrices into one
          // Or masking?
          Broadcast(Tanh(bpp::_1 + bpp::_2), Temp1_, Temp2_);
          Prod(A_, w_.V_, Temp1_, false, true);
          size_t words = SourceContext.Rows();
          // batch size, for batching, divide by numer of sentences
          size_t batchSize = HiddenState.Rows(); 
          A_.Reshape(batchSize, words); // due to broadcasting above
          Element(bpp::_1 + w_.C_(0,0), A_);
          mblas::Softmax(A_);
          
          Prod(AlignedSourceContext, A_, SourceContext);
        }
        
        void GetAttention(mblas::Matrix& Attention) {
          mblas::Copy(Attention, A_);
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
        : w_(model),
        blazeW1_(w_.W1_),
        blazeW2_(w_.W2_),
        blazeW3_(w_.W3_),
        blazeW4_(w_.W4_),
        filtered_(false)
        {}
          
        void GetProbs(mblas::Matrix& Probs,
                  const mblas::Matrix& State,
                  const mblas::Matrix& Embedding,
                  const mblas::Matrix& AlignedSourceContext) {
          using namespace mblas;
          
          T1_ = State * blazeW1_;
          T2_ = Embedding * blazeW2_;
          T3_ = AlignedSourceContext * blazeW3_;
          
          BroadcastVec(bpp::_1 + bpp::_2, T1_, w_.B1_);
          BroadcastVec(bpp::_1 + bpp::_2, T2_, w_.B2_);
          BroadcastVec(bpp::_1 + bpp::_2, T3_, w_.B3_);
      
          Element(Tanh(bpp::_1 + bpp::_2 + bpp::_3), T1_, T2_, T3_);
          
          DynMatrix T1 = T1_;
          if(!filtered_) {
            Probs = T1 * blazeW4_;
            //Prod(Probs, T1_, w_.W4_);
            BroadcastVec(bpp::_1 + bpp::_2, Probs, w_.B4_);
          } else {
            Probs = T1 * FilteredW4_;
            BroadcastVec(bpp::_1 + bpp::_2, Probs, FilteredB4_);
          }
          mblas::SoftmaxLog(Probs);
        }
    
        void Filter(const std::vector<size_t>& ids) {
          filtered_ = true;
          
          using namespace mblas;
          AssembleCols(FilteredW4_, w_.W4_, ids);
          AssembleCols(FilteredB4_, w_.B4_, ids);
        }
       
      private:        
        const Weights& w_;
        
        blaze::DynamicMatrix<float, blaze::rowMajor> blazeW1_;
        blaze::DynamicMatrix<float, blaze::rowMajor> blazeW2_;
        blaze::DynamicMatrix<float, blaze::rowMajor> blazeW3_;
        blaze::DynamicMatrix<float, blaze::rowMajor> blazeW4_;
        
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
      Embedding.Clear();
      Embedding.Resize(batchSize, embeddings_.GetCols(), 0);
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
