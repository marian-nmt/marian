#pragma once

#include "layers/logits.h"  // @HACK: for factored embeddings only so far
#include "marian.h"
#include "rnn/types.h"

namespace marian {

class EncoderState {
private:
  Expr context_;
  Expr mask_;  // [beam depth=1, max length, batch size, vector dim=1] source mask
  Ptr<data::CorpusBatch> batch_;

public:
  EncoderState(Expr context, Expr mask, Ptr<data::CorpusBatch> batch)
      : context_(context), mask_(mask), batch_(batch) {}

  EncoderState() {}
  virtual ~EncoderState() {}

  virtual Expr getContext() const { return context_; }
  virtual Expr getAttended() const { return context_; }
  virtual Expr getMask() const {
    return mask_;
  }  // source batch mask; may have additional positions suppressed

  virtual const Words& getSourceWords() { return batch_->front()->data(); }

  // Sub-select active batch entries from encoder context and context mask
  Ptr<EncoderState> select(
      const std::vector<IndexType>& batchIndices) {  // [batchIndex] indices of active batch entries
    // Dimension -2 is OK for both, RNN and Transformer models as the encoder context in Transformer
    // gets transposed to the same dimension layout
    return New<EncoderState>(
        index_select(context_, -2, batchIndices), index_select(mask_, -2, batchIndices), batch_);
  }
};

class DecoderState {
protected:
  rnn::States states_;  // states of individual decoder layers
  Logits logProbs_;
  std::vector<Ptr<EncoderState>> encStates_;
  Ptr<data::CorpusBatch> batch_;

  Expr targetHistoryEmbeddings_;  // decoder history (teacher-forced or from decoding), embedded
  Expr targetMask_;
  Words targetWords_;  // target labels

  // Keep track of current target token position during translation
  size_t position_{0};

public:
  DecoderState(const rnn::States& states,
               Logits logProbs,
               const std::vector<Ptr<EncoderState>>& encStates,
               Ptr<data::CorpusBatch> batch)
      : states_(states), logProbs_(logProbs), encStates_(encStates), batch_(batch) {}
  virtual ~DecoderState() {}

  // @TODO: Do we need all these to be virtual?
  virtual const std::vector<Ptr<EncoderState>>& getEncoderStates() const { return encStates_; }

  virtual Logits getLogProbs() const { return logProbs_; }
  virtual void setLogProbs(Logits logProbs) { logProbs_ = logProbs; }

  // @TODO: should this be a constructor? Then derived classes can call this without the New<> in
  // the loop
  virtual Ptr<DecoderState> select(
      const std::vector<IndexType>& hypIndices,    // [beamIndex * activeBatchSize + batchIndex]
      const std::vector<IndexType>& batchIndices,  // [batchIndex]
      int beamSize) const {
    std::vector<Ptr<EncoderState>> newEncStates;
    for(auto& es : encStates_)
      // If the size of the batch dimension of the encoder state context changed, subselect the
      // correct batch entries
      newEncStates.push_back(
          es->getContext()->shape()[-2] == batchIndices.size() ? es : es->select(batchIndices));

    // hypindices matches batchIndices in terms of batch dimension, so we only need hypIndices
    auto selectedState
        = New<DecoderState>(states_.select(hypIndices, beamSize, /*isBatchMajor=*/false),
                            logProbs_,
                            newEncStates,
                            batch_);

    // Set positon of new state based on the target token position of current state
    selectedState->setPosition(getPosition());
    return selectedState;
  }

  virtual const rnn::States& getStates() const { return states_; }

  virtual Expr getTargetHistoryEmbeddings() const { return targetHistoryEmbeddings_; };
  virtual void setTargetHistoryEmbeddings(Expr targetHistoryEmbeddings) {
    targetHistoryEmbeddings_ = targetHistoryEmbeddings;
  }

  virtual const Words& getTargetWords() const { return targetWords_; };
  virtual void setTargetWords(const Words& targetWords) { targetWords_ = targetWords; }

  virtual Expr getTargetMask() const { return targetMask_; };
  virtual void setTargetMask(Expr targetMask) { targetMask_ = targetMask; }

  virtual const Words& getSourceWords() const { return getEncoderStates()[0]->getSourceWords(); }

  Ptr<data::CorpusBatch> getBatch() const { return batch_; }

  // Set current target token position in state when decoding
  size_t getPosition() const { return position_; }

  // Set current target token position in state when decoding
  void setPosition(size_t position) { position_ = position; }

  virtual void blacklist(Expr /*totalCosts*/, Ptr<data::CorpusBatch> /*batch*/) {}
};

/**
 * Classifier output based on DecoderState
 * @TODO: should be unified with DecoderState or not be used at all as Classifier do not really have
 * stateful output.
 */
class ClassifierState {
private:
  Expr logProbs_;
  std::vector<Ptr<EncoderState>> encStates_;
  Ptr<data::CorpusBatch> batch_;

  Expr targetMask_;
  Words targetWords_;

public:
  virtual ~ClassifierState() {}
  virtual Expr getLogProbs() const { return logProbs_; }
  virtual void setLogProbs(Expr logProbs) { logProbs_ = logProbs; }

  virtual const Words& getTargetWords() const { return targetWords_; };
  virtual void setTargetWords(const Words& targetWords) { targetWords_ = targetWords; }

  virtual Expr getTargetMask() const { return targetMask_; };

  virtual void setTargetMask(Expr targetMask) { targetMask_ = targetMask; }
};

}  // namespace marian
