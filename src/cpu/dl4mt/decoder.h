#pragma once

#include "../mblas/matrix.h"
#include "model.h"
#include "gru.h"

namespace CPU
{

class Decoder {
  private:
    template <class Weights>
    class Embeddings {
      public:
        Embeddings(const Weights& model);
            
        void Lookup(mblas::Matrix& Rows, const std::vector<size_t>& ids);
        
        size_t GetCols();

        size_t GetRows() const;

      private:
        const Weights& w_;
    };
    
    //////////////////////////////////////////////////////////////
    template <class Weights1, class Weights2>
    class RNNHidden {
      public:
        RNNHidden(const Weights1& initModel, const Weights2& gruModel);
        
        void InitializeState(mblas::Matrix& State,
                             const mblas::Matrix& SourceContext,
                             const size_t batchSize = 1);
        
        void GetNextState(mblas::Matrix& NextState,
                          const mblas::Matrix& State,
                          const mblas::Matrix& Context);

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
        RNNFinal(const Weights& model);
        
        void GetNextState(mblas::Matrix& NextState,
                          const mblas::Matrix& State,
                          const mblas::Matrix& Context);
      private:
        const GRU<Weights> gru_;
    };
        
    //////////////////////////////////////////////////////////////
    template <class Weights>
    class Attention {
      public:
        Attention(const Weights& model);
          
        void GetAlignedSourceContext(mblas::Matrix& AlignedSourceContext,
                                     const mblas::Matrix& HiddenState,
                                     const mblas::Matrix& SourceContext);
        
        void GetAttention(mblas::Matrix& Attention);

      private:
        const Weights& w_;
        
        mblas::Matrix Temp1_;
        mblas::Matrix Temp2_;
        mblas::Matrix A_;
        mblas::ColumnVector V_;
    };
    
    //////////////////////////////////////////////////////////////
    template <class Weights>
    class Softmax {
      public:
        Softmax(const Weights& model);
          
        void GetProbs(mblas::ArrayMatrix& Probs,
                  const mblas::Matrix& State,
                  const mblas::Matrix& Embedding,
                  const mblas::Matrix& AlignedSourceContext);
    
        void Filter(const std::vector<size_t>& ids);
       
      private:        
        const Weights& w_;
        bool filtered_;
        
        mblas::Matrix FilteredW4_;
        mblas::Matrix FilteredB4_;
        
        mblas::Matrix T1_;
        mblas::Matrix T2_;
        mblas::Matrix T3_;
        mblas::Matrix Probs_;
    };
    
  public:
    Decoder(const Weights& model);
    
    void MakeStep(mblas::Matrix& NextState,
                  mblas::ArrayMatrix& Probs,
                  const mblas::Matrix& State,
                  const mblas::Matrix& Embeddings,
                  const mblas::Matrix& SourceContext);
    
    void EmptyState(mblas::Matrix& State,
                    const mblas::Matrix& SourceContext,
                    size_t batchSize = 1);
    
    void EmptyEmbedding(mblas::Matrix& Embedding,
                        size_t batchSize = 1);
    
    void Lookup(mblas::Matrix& Embedding,
                const std::vector<size_t>& w);
    
    void Filter(const std::vector<size_t>& ids);
      
    void GetAttention(mblas::Matrix& attention);
    
    size_t GetVocabSize() const;
    
  private:
    
    void GetHiddenState(mblas::Matrix& HiddenState,
                        const mblas::Matrix& PrevState,
                        const mblas::Matrix& Embedding);
    
    void GetAlignedSourceContext(mblas::Matrix& AlignedSourceContext,
                                 const mblas::Matrix& HiddenState,
                                 const mblas::Matrix& SourceContext);
    
    void GetNextState(mblas::Matrix& State,
                      const mblas::Matrix& HiddenState,
                      const mblas::Matrix& AlignedSourceContext);
    
    void GetProbs(mblas::ArrayMatrix& Probs,
                  const mblas::Matrix& State,
                  const mblas::Matrix& Embedding,
                  const mblas::Matrix& AlignedSourceContext);
    
  private:
    mblas::Matrix HiddenState_;
    mblas::Matrix AlignedSourceContext_;  
    
    Embeddings<Weights::Embeddings> embeddings_;
    RNNHidden<Weights::DecInit, Weights::GRU> rnn1_;
    RNNFinal<Weights::DecGRU2> rnn2_;
    Attention<Weights::DecAttention> attention_;
    Softmax<Weights::DecSoftmax> softmax_;
};

}

