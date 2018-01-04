#pragma once
#include <memory>
#include <vector>
#include <stddef.h>
#include <cassert>
#include "god.h"
#include "history.h"

namespace amunmt {

class Sentences;
class History;
class BufferOutput;

class Hypothesis;
using HypothesisPtr = std::shared_ptr<Hypothesis>;
using Hypotheses = std::vector<HypothesisPtr>;
using HypothesesBatch = std::vector<Hypotheses>;

//////////////////////////////////////////////////////////////////////////////////////////
class HistoriesElement
{
public:
  HistoriesElement(const SentencePtr &sentence, bool normalizeScore);

  unsigned GetBeamSize() const
  { return beamSize_; }

  void SetNewBeamSize(unsigned val);

  const History &GetHistory() const
  { return history_; }

  const SentencePtr &GetSentence() const
  { return sentence_; }

  const Hypotheses &GetHypotheses() const
  { return hypos_; }

  Hypotheses &GetHypotheses()
  { return hypos_; }

  void Add();

  bool IsFirst() const;

  void StartCalcBeam();

protected:
  unsigned beamSize_;  // beam size 0..beam
  History history_;
  SentencePtr sentence_;
  Hypotheses hypos_;

};

///////////////////////////////////////////////////////////////////////////////////////////////////
using HistoriesElementPtr = std::shared_ptr<HistoriesElement>;
///////////////////////////////////////////////////////////////////////////////////////////////////

class Histories
{
public:
  Histories(bool normalizeScore);
  Histories() = delete;
  Histories(const Histories&) = delete;

  void Init(const std::vector<BufferOutput> &newSentences);

  size_t size() const
  { return coll_.size(); }

  unsigned NumActive() const
  { return active_; }

  bool NormalizeScore() const
  { return normalizeScore_; }

  const HistoriesElementPtr &Get(size_t ind) const
  { return coll_[ind]; }

  HistoriesElementPtr &Get(size_t ind)
  { return coll_[ind]; }

  void Set(size_t ind, HistoriesElement *val);

  size_t GetBeamSize(size_t ind) const;
  bool Empty(size_t ind) const;

  size_t MaxLength() const;

  void SetNewBeamSize(unsigned val);

  std::vector<unsigned> GetBeamSizes() const;
  size_t GetTotalBeamSize() const;
  size_t NumCandidates() const;
  std::vector<char> IsFirsts() const;

  void Add(const God &god);

  Hypotheses GetSurvivors() const;
  std::vector<unsigned> GetWords() const;
  std::vector<unsigned> GetPrevStateIndices() const;

  std::vector<unsigned> Hypo2Batch() const;

  void OutputAll(const God &god);

  void StartCalcBeam();

  virtual std::string Debug(size_t verbosity = 1) const;

  // topup
  void StartTopup();

  void Topup(HistoriesElement *val);

  const std::vector<unsigned> &GetNewBatchIds() const
  { return newBatchIds_; }

  const std::vector<unsigned> &GetNewSentenceLengths() const;

  void BatchIds(std::vector<unsigned> &newBatch, std::vector<unsigned> &oldBatch) const;

protected:
  bool normalizeScore_;
  std::vector<HistoriesElementPtr> coll_;
  unsigned active_;

  // topup
  std::vector<unsigned> newBatchIds_;
  unsigned nextBatchInd_;
  mutable std::vector<unsigned> newSentenceLengths_;

  unsigned FindNextEmptyIndex();

};

}
