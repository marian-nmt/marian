#pragma once

#include <yaml-cpp/yaml.h>

#include "common/histories.h"
#include "gpu/mblas/vector.h"
#include "gpu/mblas/matrix_functions.h"
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

        void Lookup(mblas::Matrix& Rows, const std::vector<unsigned>& ids)
        {
          using namespace mblas;
          std::vector<unsigned> tids = ids;
          for(unsigned &id : tids) {
            if(id >= w_.E_->dim(0)) {
              id = 1;
            }
          }
          indices_.copyFrom(tids);

          Assemble(Rows, *w_.E_, indices_);
        }

        void LookupTopup(mblas::Matrix& Rows,
                        const std::vector<unsigned>& ids,
                        const mblas::Vector<unsigned> &d_oldHypoIds,
                        unsigned numHypos)
        {
          using namespace mblas;
          //std::cerr << "ids=" << amunmt::Debug(ids, 2) << std::endl;
          //std::cerr << "d_oldHypoIds=" << d_oldHypoIds.Debug(2) << std::endl;

          assert(ids.size() == d_oldHypoIds.size());

          std::vector<unsigned> tids = ids;
          for(unsigned &id : tids) {
            if(id >= w_.E_->dim(0)) {
              id = 1;
            }
          }
          indices_.copyFrom(tids);

          AssembleTopup(Rows, *w_.E_, indices_, numHypos, d_oldHypoIds);

          //std::cerr << "Rows=" << Rows.Debug(0) << std::endl;
        }

        unsigned GetCols() const {
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
                             const mblas::Matrix &SourceContext,
                             unsigned batchSize,
                             const mblas::Vector<unsigned> &sentenceLengths) const
        {
          using namespace mblas;

          CellLength cellLength = gru_->GetStateLength();
          if (cellLength.cell > 0) {
            State.cell->NewSize(batchSize, cellLength.cell);
            mblas::Fill(*(State.cell), 0.0f);
          }

          thread_local mblas::Matrix Temp2;
          Temp2.NewSize(batchSize, SourceContext.dim(1), 1, 1);

          Mean(Temp2, SourceContext, sentenceLengths);

          TIME_CMD("Prod1", Prod(*(State.output), Temp2, *w_.Wi_));

          if (w_.Gamma_->size()) {
            Normalization(*(State.output), *(State.output), *w_.Gamma_, *w_.Bi_, 1e-9);
            Element(Tanh(_1), *(State.output));
          } else {
            BroadcastVec(Tanh(_1 + _2), *(State.output), *w_.Bi_);
          }
        }

        void InitializeStateTopup(CellState& State,
                                  const std::vector<BufferOutput> &newSentences,
                                  const std::vector<unsigned> &newHypoIds) const
        {
          mblas::AddNewStates(State, newHypoIds, newSentences);

        }

        void GetNextState(CellState& NextState,
                          const CellState& State,
                          const mblas::Matrix& Context) const
        {
          gru_->GetNextState(NextState, State, Context);
        }

      private:
        const Weights& w_;
        std::unique_ptr<Cell> gru_;

        RNNHidden(const RNNHidden&) = delete;
    };

    class RNNFinal {
      public:
        RNNFinal(std::unique_ptr<Cell> cell)
          : gru_(std::move(cell)) {}

        void GetNextState(CellState& NextState,
                          const CellState& State,
                          const mblas::Matrix& Context) const
        {
          gru_->GetNextState(NextState, State, Context);
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
          , dHypo2Batch_(god.Get<unsigned>("mini-batch") * god.Get<unsigned>("beam-size"), 0)
        {}

        void Init(const mblas::Matrix& SourceContext,
                  mblas::Matrix& SCU) const
        {
          using namespace mblas;

          TIME_CMD("Prod2", Prod(/*h_[0],*/ SCU, SourceContext, *w_.U_));
          if (w_.Gamma_1_->size()) {
            Normalization(SCU, SCU, *w_.Gamma_1_, *w_.B_, 1e-9);
          }
        }

        void InitTopup(const mblas::Matrix& SourceContext,
                  mblas::Matrix& SCU,
                  const std::vector<BufferOutput> &newSentences,
                  const mblas::Vector<unsigned> &d_oldBatchIds,
                  const std::vector<unsigned> &newBatchIds
                  ) const
        {
          using namespace mblas;

          unsigned maxLength = maxLength = SourceContext.dim(0);
          ResizeMatrix3(SCU, {0, maxLength}, d_oldBatchIds);
          AddNewSCU(SCU, newBatchIds, newSentences);
        }

        void GetAlignedSourceContext(mblas::Matrix& AlignedSourceContext,
                                     const CellState& HiddenState,
                                     const Histories& histories,
                                     const mblas::Matrix& SourceContext,
                                     const mblas::Matrix& SCU,
                                     const mblas::Vector<unsigned> &sentenceLengths)
        {
          // mapping = 1/0 whether each position, in each sentence in the batch is actually a valid word
          // hypo2Batch = which sentence is each element in the batch. eg 0 0 1 2 2 2 = first 2 belongs to sent0, 3rd is sent1, 4th and 5th is sent2
          // dHypo2Batch_ = fixed length (batch*beam) version of hypo2Batch

          using namespace mblas;
          BEGIN_TIMER("GetAlignedSourceContext");

          unsigned maxLength = SourceContext.dim(0);
          unsigned batchSize = SourceContext.dim(3);

          std::vector<unsigned> hypo2Batch = histories.Hypo2Batch();
          dHypo2Batch_.copyFrom(hypo2Batch);

          TIME_CMD("Prod3", Prod(/*h_[1],*/ Temp2_, *(HiddenState.output), *w_.W_));

          if (w_.Gamma_2_->size()) {
            Normalization(Temp2_, Temp2_, *w_.Gamma_2_, 1e-9);
          } else {
            BroadcastVec(_1 + _2, Temp2_, *w_.B_/*, s_[1]*/);
          }

          Broadcast(Tanh(_1 + _2), Temp1_, SCU, Temp2_, dHypo2Batch_, maxLength);

          TIME_CMD("Prod4", Prod(A_, *w_.V_, Temp1_, true));

          BEGIN_TIMER("Softmax");
          mblas::Softmax(A_, dHypo2Batch_, sentenceLengths, batchSize);
          PAUSE_TIMER("Softmax");

          BEGIN_TIMER("WeightedMean");
          mblas::WeightedMean(AlignedSourceContext, A_, SourceContext, dHypo2Batch_);
          PAUSE_TIMER("WeightedMean");

          PAUSE_TIMER("GetAlignedSourceContext");
        }

        void GetAttention(mblas::Matrix& Attention) {
          mblas::Copy(Attention, A_);
        }

        mblas::Matrix& GetAttention() {
          return A_;
        }

      private:
        const Weights& w_;

        mblas::Vector<unsigned> dHypo2Batch_;

        mblas::Matrix Temp1_;
        mblas::Matrix Temp2_;
        mblas::Matrix A_;

        mblas::Matrix Ones_;
        mblas::Matrix Sums_;

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

        void GetProbs(mblas::Matrix& Probs,
            std::shared_ptr<mblas::Matrix> &b4,
                  const CellState& State,
                  const mblas::Matrix& Embedding,
                  const mblas::Matrix& AlignedSourceContext,
                  bool useFusedSoftmax)
        {
          using namespace mblas;

          //BEGIN_TIMER("GetProbs.Prod");
          TIME_CMD("Prod5", Prod(/*h_[0],*/ T1_, *(State.output), *w_.W1_));
          //PAUSE_TIMER("GetProbs.Prod");

          //BEGIN_TIMER("GetProbs.Normalization/BroadcastVec");
          if (w_.Gamma_1_->size()) {
            Normalization(T1_, T1_, *w_.Gamma_1_, *w_.B1_, 1e-9);
          } else {
            BroadcastVec(_1 + _2, T1_, *w_.B1_ /*,s_[0]*/);
          }
          //PAUSE_TIMER("GetProbs.Normalization/BroadcastVec");

          //BEGIN_TIMER("GetProbs.Prod2");
          TIME_CMD("Prod6", Prod(/*h_[1],*/ T2_, Embedding, *w_.W2_));
          //PAUSE_TIMER("GetProbs.Prod2");

          //BEGIN_TIMER("GetProbs.Normalization/BroadcastVec2");
          if (w_.Gamma_0_->size()) {
            Normalization(T2_, T2_, *w_.Gamma_0_, *w_.B2_, 1e-9);
          } else {
            BroadcastVec(_1 + _2, T2_, *w_.B2_ /*,s_[1]*/);
          }
          //PAUSE_TIMER("GetProbs.Normalization/BroadcastVec2");

          //BEGIN_TIMER("GetProbs.Prod3");
          TIME_CMD("Prod7", Prod(/*h_[2],*/ T3_, AlignedSourceContext, *w_.W3_));
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

          std::shared_ptr<mblas::Matrix> w4;
          if(!filtered_) {
            w4 = w_.W4_;
            b4 = w_.B4_;
          } else {
            w4.reset(&FilteredW4_);
            b4.reset(&FilteredB4_);
          }

          //BEGIN_TIMER("GetProbs.NewSize");
          Probs.NewSize(T1_.dim(0), w4->dim(1));
          //PAUSE_TIMER("GetProbs.NewSize");

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
        mblas::Matrix FilteredW4_;
        mblas::Matrix FilteredB4_;

        mblas::Matrix T1_;
        mblas::Matrix T2_;
        mblas::Matrix T3_;

        mblas::Matrix TempW4;
        mblas::Matrix TempB4;

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
                const mblas::Matrix& Embeddings,
                const Histories& histories,
                bool useFusedSoftmax,
                const mblas::Matrix& SourceContext,
                const mblas::Matrix& SCU,
                const mblas::Vector<unsigned> &sentenceLengths)
    {
      //BEGIN_TIMER("Decode");

      GetHiddenState(HiddenState_, State, Embeddings);
      GetAlignedSourceContext(AlignedSourceContext_,
                              HiddenState_,
                              histories,
                              SourceContext,
                              SCU,
                              sentenceLengths);
      GetNextState(NextState, HiddenState_, AlignedSourceContext_);
      GetProbs(NextState, Embeddings, AlignedSourceContext_, useFusedSoftmax);

      //PAUSE_TIMER("Decode");
    }

    mblas::Matrix& GetProbs() {
      return Probs_;
    }

    void EmptyStateTopup(CellState& State,
                    const mblas::Matrix &SourceContext,
                    mblas::Matrix& SCU,
                    const std::vector<BufferOutput> &newSentences,
                    const mblas::Vector<unsigned> &d_oldBatchIds,
                    const std::vector<unsigned> &newBatchIds,
                    const std::vector<unsigned> &newHypoIds) const
    {
      rnn1_.InitializeStateTopup(State, newSentences, newHypoIds);
      alignment_.InitTopup(SourceContext, SCU, newSentences, d_oldBatchIds, newBatchIds);
    }

    void EmptyEmbedding(mblas::Matrix& Embedding, unsigned batchSize) const
    {
      Embedding.NewSize(batchSize, embeddings_.GetCols());
      mblas::Fill(Embedding, 0);
    }

    void EmptyEmbeddingTopup(mblas::Matrix& Embedding,
                            unsigned totalBeamSize,
                            const mblas::Vector<unsigned> &d_newHypoIds) const
    {
      mblas::Fill0(Embedding, 0, d_newHypoIds);
    }

    void Lookup(mblas::Matrix& Embedding,
                const std::vector<unsigned>& w) {
      embeddings_.Lookup(Embedding, w);
    }

    void LookupTopup(mblas::Matrix& Embedding,
                const std::vector<unsigned>& w,
                const Histories &histories,
                const mblas::Vector<unsigned> &d_oldHypoIds)
    {
      embeddings_.LookupTopup(Embedding, w, d_oldHypoIds, histories.GetTotalBeamSize());
    }

    void Filter(const std::vector<unsigned>& ids) {
      softmax_.Filter(ids);
    }

    void GetAttention(mblas::Matrix& Attention) {
      alignment_.GetAttention(Attention);
    }

    unsigned GetVocabSize() const {
      return embeddings_.GetRows();
    }

    mblas::Matrix& GetAttention() {
      return alignment_.GetAttention();
    }

    mblas::Vector<NthOutBatch>& GetNBest() {
      return nBest_;
    }

    const mblas::Matrix *GetBias() const {
      return b4_.get();
    }

    Alignment<Weights::DecAlignment> &GetAligner()
    { return alignment_; }

    RNNHidden<Weights::DecInit> &GetHiddenRNN()
    { return rnn1_; }

  private:

    void GetHiddenState(CellState& HiddenState,
                        const CellState& PrevState,
                        const mblas::Matrix& Embedding) {
      rnn1_.GetNextState(HiddenState, PrevState, Embedding);
    }

    void GetAlignedSourceContext(mblas::Matrix& AlignedSourceContext,
                                 const CellState& HiddenState,
                                 const Histories& histories,
                                 const mblas::Matrix& SourceContext,
                                 const mblas::Matrix& SCU,
                                 const mblas::Vector<unsigned> &sentenceLengths)

    {
      alignment_.GetAlignedSourceContext(AlignedSourceContext,
                                        HiddenState,
                                        histories,
                                        SourceContext,
                                        SCU,
                                        sentenceLengths);

    }

    void GetNextState(CellState& State,
                      const CellState& HiddenState,
                      const mblas::Matrix& AlignedSourceContext) const
    {
      rnn2_.GetNextState(State, HiddenState, AlignedSourceContext);
    }


    void GetProbs(const CellState& State,
                  const mblas::Matrix& Embedding,
                  const mblas::Matrix& AlignedSourceContext,
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
    mblas::Matrix AlignedSourceContext_;
    mblas::Matrix Probs_;

    Embeddings<Weights::DecEmbeddings> embeddings_;
    RNNHidden<Weights::DecInit> rnn1_;
    RNNFinal rnn2_;
    Alignment<Weights::DecAlignment> alignment_;
    Softmax<Weights::DecSoftmax> softmax_;

    mblas::Vector<NthOutBatch> nBest_;
    std::shared_ptr<mblas::Matrix> b4_;

    Decoder(const Decoder&) = delete;
};

}
}

