#pragma once

#include <sstream>
#include <yaml-cpp/yaml.h>

#include "gpu/mblas/vector.h"
#include "gpu/mblas/tensor_functions.h"
#include "model.h"
#include "gru.h"
#include "lstm.h"
#include "gpu/types-gpu.h"
#include "common/god.h"
#include "cell.h"
#include "cellstate.h"

namespace amunmt {
namespace GPU {

class Decoder {
  private:
    template <class Weights>
    class Embeddings {
      public:
        Embeddings(const Weights& model)
        : w_(model)
        {}

        void Lookup(mblas::Tensor& Rows, const std::vector<unsigned>& ids) {
          using namespace mblas;
          std::vector<unsigned> tids = ids;
          for(auto&& id : tids)
            if(id >= w_.E_->dim(0))
              id = 1;
          indices_.newSize(tids.size());

          mblas::copy(tids.data(),
              tids.size(),
              indices_.data(),
              cudaMemcpyHostToDevice);

          Assemble(Rows, *w_.E_, indices_);
        }

        unsigned GetCols() {
          return w_.E_->dim(1);
        }

        unsigned GetRows() const {
          return w_.E_->dim(0);
        }

      private:
        const Weights& w_;
        mblas::Vector<unsigned> indices_;

        Embeddings(const Embeddings&) = delete;
    };

    template <class Weights>
    class RNNHidden {
      public:
        RNNHidden() = default;
        RNNHidden(const Weights& initModel, std::unique_ptr<Cell> cell)
        : w_(initModel)
        , gru_(std::move(cell))
        {}

        void InitializeState(CellState& State,
                             const mblas::Tensor& SourceContext,
                             const unsigned batchSize,
                             const mblas::Vector<unsigned> &sentenceLengths)
        {
          using namespace mblas;

          CellLength cellLength = gru_->GetStateLength();
          if (cellLength.cell > 0) {
            State.cell->NewSize(batchSize, cellLength.cell);
            mblas::Fill(*(State.cell), 0.0f);
          }
          //std::cerr << "1Temp2_=" << Temp2_.Debug(1) << std::endl;
          Temp2_.NewSize(batchSize, SourceContext.dim(1), 1, 1);
          //std::cerr << "2Temp2_=" << Temp2_.Debug(1) << std::endl;

          //std::cerr << "SourceContext=" << SourceContext.Debug(1) << std::endl;
          //std::cerr << "mapping=" << Debug(mapping, 2) << std::endl;
          Mean(Temp2_, SourceContext, sentenceLengths);

          //std::cerr << "1State=" << State.Debug(1) << std::endl;
          //std::cerr << "3Temp2_=" << Temp2_.Debug(1) << std::endl;
          //std::cerr << "w_.Wi_=" << w_.Wi_->Debug(1) << std::endl;
          Prod(*(State.output), Temp2_, *w_.Wi_);

          //std::cerr << "2State=" << State.Debug(1) << std::endl;
          //State.ReduceDimensions();

          if (w_.Gamma_->size()) {
            Normalization(*(State.output), *(State.output), *w_.Gamma_, *w_.Bi_, 1e-9);
            Element(Tanh(_1), *(State.output));
          } else {
            BroadcastVec(Tanh(_1 + _2), *(State.output), *w_.Bi_);
          }
          //std::cerr << "3State=" << State.Debug(1) << std::endl;
          //std::cerr << "\n";
        }

        void GetNextState(CellState& NextState,
                          const CellState& State,
                          const mblas::Tensor& Context) {
          gru_->GetNextState(NextState, State, Context);
        }

      private:
        const Weights& w_;
        std::unique_ptr<Cell> gru_;

        mblas::Tensor Temp2_;

        RNNHidden(const RNNHidden&) = delete;
    };

    class RNNFinal {
      public:
        RNNFinal(std::unique_ptr<Cell> cell)
          : gru_(std::move(cell)) {}

        void GetNextState(CellState& NextState,
                          const CellState& State,
                          const mblas::Tensor& Context) {
          gru_->GetNextState(NextState, State, Context);
        }

        std::string Debug(unsigned verbosity = 1) const
        {
          std::stringstream strm;
          strm << gru_->Debug(verbosity);

          return strm.str();
        }

      private:
        std::unique_ptr<Cell> gru_;

        RNNFinal(const RNNFinal&) = delete;
    };

    template <class Weights>
    class Alignment {
      public:
        Alignment(const God &god, const Weights& model)
          : w_(model)
          , dBatchMapping_(god.Get<unsigned>("mini-batch") * god.Get<unsigned>("beam-size"), 0)
        {}

        void Init(const mblas::Tensor& SourceContext) {
          using namespace mblas;

          Prod(/*h_[0],*/ SCU_, SourceContext, *w_.U_);
          //std::cerr << "SCU_=" << SCU_.Debug(1) << std::endl;

          if (w_.Gamma_1_->size()) {
            Normalization(SCU_, SCU_, *w_.Gamma_1_, *w_.B_, 1e-9);
          }
        }

        unsigned GetMaxLength(const std::vector<unsigned>& h_sentenceLengths, const std::vector<unsigned>& beamSizes) const
        {
          assert(h_sentenceLengths.size() == beamSizes.size());

          unsigned ret = 0;
          for (unsigned i = 0; i < beamSizes.size(); ++i) {
            if (beamSizes[i]) {
              if (ret < h_sentenceLengths[i]) {
                ret = h_sentenceLengths[i];
              }
            }
          }
          return ret;
        }

        void GetAlignedSourceContext(mblas::Tensor& AlignedSourceContext,
                                     const CellState& HiddenState,
                                     const mblas::Tensor& SourceContext,
                                     const std::vector<unsigned>& h_sentenceLengths,
                                     const mblas::Vector<unsigned> &sentenceLengths,
                                     const std::vector<unsigned>& beamSizes)
        {
          // mapping = 1/0 whether each position, in each sentence in the batch is actually a valid word
          // batchMapping = which sentence is each element in the batch. eg 0 0 1 2 2 2 = first 2 belongs to sent0, 3rd is sent1, 4th and 5th is sent2
          // dBatchMapping_ = fixed length (batch*beam) version of dBatchMapping_

          using namespace mblas;
          BEGIN_TIMER("GetAlignedSourceContext");

          unsigned maxLength = SourceContext.dim(0);
          unsigned batchSize = SourceContext.dim(3);
          //std::cerr << "batchSize=" << batchSize << std::endl;
          //std::cerr << "HiddenState=" << HiddenState.Debug(0) << std::endl;
          //unsigned maxLength = GetMaxLength(h_sentenceLengths, beamSizes);
          /*
          std::cerr << "SourceContext=" << SourceContext.Debug(0) << std::endl;
          std::cerr << "beamSizes=" << Debug(beamSizes, 2) << std::endl;
          std::cerr << "maxLength=" << SourceContext.dim(0) << " " << maxLength << std::endl;
          */

          std::vector<unsigned> batchMapping(HiddenState.output->dim(0));
          unsigned k = 0;
          for (unsigned i = 0; i < beamSizes.size(); ++i) {
            for (unsigned j = 0; j < beamSizes[i]; ++j) {
              batchMapping[k++] = i;
            }
          }

          dBatchMapping_.newSize(batchMapping.size());
          mblas::copy(batchMapping.data(),
              batchMapping.size(),
              dBatchMapping_.data(),
              cudaMemcpyHostToDevice);

          /*
          std::cerr << "SourceContext=" << SourceContext.Debug(0) << std::endl;
          std::cerr << "AlignedSourceContext=" << AlignedSourceContext.Debug(0) << std::endl;
          std::cerr << "A_=" << A_.Debug(0) << std::endl;
          std::cerr << "sentenceLengths=" << sentenceLengths.Debug(2) << std::endl;
          */

          Prod(/*h_[1],*/ Temp2_, *(HiddenState.output), *w_.W_);
          //std::cerr << "1Temp2_=" << Temp2_.Debug() << std::endl;

          if (w_.Gamma_2_->size()) {
            Normalization(Temp2_, Temp2_, *w_.Gamma_2_, 1e-9);
          } else {
            BroadcastVec(_1 + _2, Temp2_, *w_.B_/*, s_[1]*/);
          }
          //std::cerr << "2Temp2_=" << Temp2_.Debug() << std::endl;

          Broadcast(Tanh(_1 + _2), Temp1_, SCU_, Temp2_, dBatchMapping_, maxLength);

          Prod(A_, *w_.V_, Temp1_, true);


          mblas::Softmax(A_, dBatchMapping_, sentenceLengths);

          mblas::WeightedMean(AlignedSourceContext, A_, SourceContext, dBatchMapping_);

          /*
          std::cerr << "AlignedSourceContext=" << AlignedSourceContext.Debug() << std::endl;
          std::cerr << "A_=" << A_.Debug() << std::endl;
          std::cerr << "SourceContext=" << SourceContext.Debug() << std::endl;
          std::cerr << "mapping=" << Debug(mapping, 2) << std::endl;
          std::cerr << "dBatchMapping_=" << Debug(dBatchMapping_, 2) << std::endl;
          std::cerr << std::endl;
          */
          PAUSE_TIMER("GetAlignedSourceContext");
        }

        void GetAttention(mblas::Tensor& Attention) {
          mblas::Copy(Attention, A_);
        }

        mblas::Tensor& GetAttention() {
          return A_;
        }

      private:
        const Weights& w_;

        mblas::Vector<unsigned> dBatchMapping_;

        mblas::Tensor SCU_;
        mblas::Tensor Temp1_;
        mblas::Tensor Temp2_;
        mblas::Tensor A_;

        mblas::Tensor Ones_;
        mblas::Tensor Sums_;

        Alignment(const Alignment&) = delete;
    };

    template <class Weights>
    class Softmax {
      public:
        Softmax(const Weights& model)
        : w_(model), filtered_(false)
        {
          mblas::Transpose(TempW4, *w_.W4_);
          mblas::Transpose(TempB4, *w_.B4_);
        }

        unsigned TensorCoreSize(unsigned origSize) {
          unsigned remainder = origSize % 8;
          if (remainder) {
            unsigned add = 8 - remainder;
            return origSize + add;
          }
          else {
            return origSize;
          }
        }

        void GetProbs(mblas::Tensor& Probs,
            std::shared_ptr<mblas::Tensor> &b4,
                  const CellState& State,
                  const mblas::Tensor& Embedding,
                  const mblas::Tensor& AlignedSourceContext,
                  bool useFusedSoftmax)
        {
          using namespace mblas;

          //BEGIN_TIMER("GetProbs.Prod");
          Prod(/*h_[0],*/ T1_, *(State.output), *w_.W1_);
          //PAUSE_TIMER("GetProbs.Prod");

          //BEGIN_TIMER("GetProbs.Normalization/BroadcastVec");
          if (w_.Gamma_1_->size()) {
            Normalization(T1_, T1_, *w_.Gamma_1_, *w_.B1_, 1e-9);
          } else {
            BroadcastVec(_1 + _2, T1_, *w_.B1_ /*,s_[0]*/);
          }
          //PAUSE_TIMER("GetProbs.Normalization/BroadcastVec");

          //BEGIN_TIMER("GetProbs.Prod2");
          Prod(/*h_[1],*/ T2_, Embedding, *w_.W2_);
          //PAUSE_TIMER("GetProbs.Prod2");

          //BEGIN_TIMER("GetProbs.Normalization/BroadcastVec2");
          if (w_.Gamma_0_->size()) {
            Normalization(T2_, T2_, *w_.Gamma_0_, *w_.B2_, 1e-9);
          } else {
            BroadcastVec(_1 + _2, T2_, *w_.B2_ /*,s_[1]*/);
          }
          //PAUSE_TIMER("GetProbs.Normalization/BroadcastVec2");

          //BEGIN_TIMER("GetProbs.Prod3");
          Prod(/*h_[2],*/ T3_, AlignedSourceContext, *w_.W3_);
          //PAUSE_TIMER("GetProbs.Prod3");

          //BEGIN_TIMER("GetProbs.Normalization/BroadcastVec3");
          if (w_.Gamma_2_->size()) {
            Normalization(T3_, T3_, *w_.Gamma_2_, *w_.B3_, 1e-9);
          } else {
            BroadcastVec(_1 + _2, T3_, *w_.B3_ /*,s_[2]*/);
          }
          //PAUSE_TIMER("GetProbs.Normalization/BroadcastVec3");

          //BEGIN_TIMER("GetProbs.Element");
          Element(Tanh(_1 + _2 + _3), T1_, T2_, T3_);
          //PAUSE_TIMER("GetProbs.Element");

          std::shared_ptr<mblas::Tensor> w4;
          if(!filtered_) {
            w4 = w_.W4_;
            b4 = w_.B4_;
          } else {
            w4.reset(&FilteredW4_);
            b4.reset(&FilteredB4_);
          }

          BEGIN_TIMER("OutputLayer");

          BEGIN_TIMER("GetProbs.Prod4");
          Prod(Probs, T1_, *w4);
          PAUSE_TIMER("GetProbs.Prod4");

          if (!useFusedSoftmax) {
            BEGIN_TIMER("GetProbs.BroadcastVec");
            BroadcastVec(_1 + _2, Probs, *b4);
            PAUSE_TIMER("GetProbs.BroadcastVec");

            BEGIN_TIMER("GetProbs.LogSoftMax");
            mblas::LogSoftmax(Probs);
            PAUSE_TIMER("GetProbs.LogSoftMax");
          }

          PAUSE_TIMER("OutputLayer");
        }

        void Filter(const std::vector<unsigned>& ids) {
          filtered_ = true;
          using namespace mblas;

          mblas::Vector<unsigned> d_ids(ids);
          Assemble(FilteredW4_, TempW4, d_ids);
          Assemble(FilteredB4_, TempB4, d_ids);

          Transpose(FilteredW4_);
          Transpose(FilteredB4_);
        }

      private:
        const Weights& w_;

        bool filtered_;
        mblas::Tensor FilteredW4_;
        mblas::Tensor FilteredB4_;

        mblas::Tensor T1_;
        mblas::Tensor T2_;
        mblas::Tensor T3_;

        mblas::Tensor TempW4;
        mblas::Tensor TempB4;

        Softmax(const Softmax&) = delete;
    };

  public:
    Decoder(const God &god, const Weights& model, const YAML::Node& config)
    : embeddings_(model.decEmbeddings_),
      rnn1_(model.decInit_, InitHiddenCell(model, config)),
      rnn2_(InitFinalCell(model, config)),
      alignment_(god, model.decAlignment_),
      softmax_(model.decSoftmax_)
    {}

    void Decode(CellState& NextState,
                  const CellState& State,
                  const mblas::Tensor& Embeddings,
                  const mblas::Tensor& SourceContext,
                  const std::vector<unsigned>& h_sentenceLengths,
                  const mblas::Vector<unsigned> &sentenceLengths,
                  const std::vector<unsigned>& beamSizes,
                  bool useFusedSoftmax)
    {
      //BEGIN_TIMER("Decode");

      //BEGIN_TIMER("GetHiddenState");

      GetHiddenState(HiddenState_, State, Embeddings);

      //HiddenState_.ReduceDimensions();
      //std::cerr << "HiddenState_=" << HiddenState_.Debug(1) << std::endl;
      //PAUSE_TIMER("GetHiddenState");

      //BEGIN_TIMER("GetAlignedSourceContext");
      GetAlignedSourceContext(AlignedSourceContext_,
                              HiddenState_,
                              SourceContext,
                              h_sentenceLengths,
                              sentenceLengths,
                              beamSizes);
      //std::cerr << "AlignedSourceContext_=" << AlignedSourceContext_.Debug(1) << std::endl;
      //PAUSE_TIMER("GetAlignedSourceContext");

      //BEGIN_TIMER("GetNextState");
      GetNextState(NextState, HiddenState_, AlignedSourceContext_);
      //PAUSE_TIMER("GetNextState");

      //BEGIN_TIMER("GetProbs");
      GetProbs(NextState, Embeddings, AlignedSourceContext_, useFusedSoftmax);
      //std::cerr << "Probs_=" << Probs_.Debug(1) << std::endl;
      //PAUSE_TIMER("GetProbs");

      //PAUSE_TIMER("Decode");
    }

    mblas::Tensor& GetProbs() {
      return Probs_;
    }

    void EmptyState(CellState& State,
                    const mblas::Tensor& SourceContext,
                    unsigned batchSize,
                    const mblas::Vector<unsigned> &sentenceLengths)
    {
      rnn1_.InitializeState(State, SourceContext, batchSize, sentenceLengths);
      alignment_.Init(SourceContext);
    }

    void EmptyEmbedding(mblas::Tensor& Embedding, unsigned batchSize = 1) {
      Embedding.NewSize(batchSize, embeddings_.GetCols());
      mblas::Fill(Embedding, 0);
    }

    void Lookup(mblas::Tensor& Embedding,
                const std::vector<unsigned>& w) {
      embeddings_.Lookup(Embedding, w);
    }

    void Filter(const std::vector<unsigned>& ids) {
      softmax_.Filter(ids);
    }

    void GetAttention(mblas::Tensor& Attention) {
      alignment_.GetAttention(Attention);
    }

    unsigned GetVocabSize() const {
      return embeddings_.GetRows();
    }

    mblas::Tensor& GetAttention() {
      return alignment_.GetAttention();
    }

    mblas::Vector<NthOutBatch>& GetNBest() {
      return nBest_;
    }

    const mblas::Tensor *GetBias() const {
      return b4_.get();
    }

  private:

    void GetHiddenState(CellState& HiddenState,
                        const CellState& PrevState,
                        const mblas::Tensor& Embedding) {
      rnn1_.GetNextState(HiddenState, PrevState, Embedding);
    }

    void GetAlignedSourceContext(mblas::Tensor& AlignedSourceContext,
                                  const CellState& HiddenState,
                                  const mblas::Tensor& SourceContext,
                                  const std::vector<unsigned>& h_sentenceLengths,
                                  const mblas::Vector<unsigned> &sentenceLengths,
                                  const std::vector<unsigned>& beamSizes)
    {
      alignment_.GetAlignedSourceContext(AlignedSourceContext,
                                        HiddenState,
                                        SourceContext,
                                        h_sentenceLengths,
                                        sentenceLengths,
                                        beamSizes);
    }

    void GetNextState(CellState& State,
                      const CellState& HiddenState,
                      const mblas::Tensor& AlignedSourceContext) {
      rnn2_.GetNextState(State, HiddenState, AlignedSourceContext);
    }


    void GetProbs(const CellState& State,
                  const mblas::Tensor& Embedding,
                  const mblas::Tensor& AlignedSourceContext,
                  bool useFusedSoftmax)
    {
      softmax_.GetProbs(Probs_, b4_, State, Embedding, AlignedSourceContext, useFusedSoftmax);
    }

    std::unique_ptr<Cell> InitHiddenCell(const Weights& model, const YAML::Node& config){
      std::string celltype = config["dec-cell"] ? config["dec-cell"].as<std::string>() : "gru";
      if (celltype == "lstm") {
        return std::unique_ptr<Cell>(new LSTM<Weights::DecLSTM1>(*(model.decLSTM1_)));
      } else if (celltype == "mlstm") {
        return std::unique_ptr<Cell>(new Multiplicative<LSTM, Weights::DecLSTM1>(*model.decMLSTM1_));
      } else if (celltype == "gru"){
        return std::unique_ptr<Cell>(new GRU<Weights::DecGRU1>(*(model.decGru1_)));
      }

      assert(false);
      return std::unique_ptr<Cell>(nullptr);
    }

    std::unique_ptr<Cell> InitFinalCell(const Weights& model, const YAML::Node& config){
      std::string hiddencell = config["dec-cell"] ? config["dec-cell"].as<std::string>() : "gru";
      std::string celltype = config["dec-cell-2"] ? config["dec-cell-2"].as<std::string>() : hiddencell;
      if (celltype == "lstm") {
        return std::unique_ptr<Cell>(new LSTM<Weights::DecLSTM2>(*(model.decLSTM2_)));
      } else if (celltype == "mlstm") {
        return std::unique_ptr<Cell>(new Multiplicative<LSTM, Weights::DecLSTM2>(*model.decMLSTM2_));
      } else if (celltype == "gru"){
        return std::unique_ptr<Cell>(new GRU<Weights::DecGRU2>(*(model.decGru2_)));
      }

      assert(false);
      return std::unique_ptr<Cell>(nullptr);
    }

  private:
    CellState HiddenState_;
    mblas::Tensor AlignedSourceContext_;
    mblas::Tensor Probs_;

    Embeddings<Weights::DecEmbeddings> embeddings_;
    RNNHidden<Weights::DecInit> rnn1_;
    RNNFinal rnn2_;
    Alignment<Weights::DecAlignment> alignment_;
    Softmax<Weights::DecSoftmax> softmax_;

    mblas::Vector<NthOutBatch> nBest_;
    std::shared_ptr<mblas::Tensor> b4_;

    Decoder(const Decoder&) = delete;
};

}
}

