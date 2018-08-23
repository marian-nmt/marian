#pragma once

#include "marian.h"
#include "rnn/types.h"

namespace marian {

class EncoderState {
private:
  Expr context_;
  Expr mask_;
  Ptr<data::CorpusBatch> batch_;

public:
  EncoderState(Expr context, Expr mask, Ptr<data::CorpusBatch> batch)
      : context_(context), mask_(mask), batch_(batch) {}

  EncoderState() {}

  virtual Expr getContext() { return context_; }
  virtual Expr getAttended() { return context_; }
  virtual Expr getMask() { return mask_; }

  virtual const std::vector<size_t>& getSourceWords() {
    return batch_->front()->data();
  }
};

class DecoderState {
protected:
  rnn::States states_; // states of individual decoder layers
  Expr logProbs_;
  std::vector<Ptr<EncoderState>> encStates_;
  Ptr<data::CorpusBatch> batch_;

  Expr targetEmbeddings_;
  Expr targetMask_;
  Expr targetIndices_;

  // Keep track of current target token position during translation
  size_t position_{0};

public:
  DecoderState(const rnn::States& states,
               Expr logProbs,
               const std::vector<Ptr<EncoderState>>& encStates,
               Ptr<data::CorpusBatch> batch)
      : states_(states), logProbs_(logProbs), encStates_(encStates), batch_(batch) {}

  // @TODO: Do we need all these to be virtual?
  virtual const std::vector<Ptr<EncoderState>>& getEncoderStates() const {
    return encStates_;
  }

  virtual Expr getLogProbs() const { return logProbs_; }
  virtual void setLogProbs(Expr logProbs) { logProbs_ = logProbs; }

  // @TODO: should this be a constructor? Then derived classes can call this without the New<> in the loop
  virtual Ptr<DecoderState> select(const std::vector<size_t>& selIdx,
                                   int beamSize) const {
    auto selectedState = New<DecoderState>(
        states_.select(selIdx, beamSize, /*isBatchMajor=*/false), logProbs_, encStates_, batch_);

    // Set positon of new state based on the target token position of current
    // state
    selectedState->setPosition(getPosition());
    return selectedState;
  }

  virtual const rnn::States& getStates() const { return states_; }

  virtual Expr getTargetEmbeddings() const { return targetEmbeddings_; };

  virtual void setTargetEmbeddings(Expr targetEmbeddings) {
    targetEmbeddings_ = targetEmbeddings;
  }

  virtual Expr getTargetIndices() const { return targetIndices_; };

  virtual void setTargetIndices(Expr targetIndices) {
    targetIndices_ = targetIndices;
  }

  virtual Expr getTargetMask() const { return targetMask_; };

  virtual void setTargetMask(Expr targetMask) { targetMask_ = targetMask; }

  virtual const std::vector<size_t>& getSourceWords() const {
    return getEncoderStates()[0]->getSourceWords();
  }

  Ptr<data::CorpusBatch> getBatch() const { return batch_; }

  // Set current target token position in state when decoding
  size_t getPosition() const { return position_; }

  // Set current target token position in state when decoding
  void setPosition(size_t position) { position_ = position; }

  virtual void blacklist(Expr totalCosts, Ptr<data::CorpusBatch> batch) {}
};
}  // namespace marian
