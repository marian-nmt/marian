#include "search.h"

#include <memory>
#include <chrono>
#include <vector>
#include <boost/iterator/permutation_iterator.hpp>

#include "decoder/god.h"
#include "decoder/sentence.h"
#include "decoder/history.h"
#include "encoder_decoder/encoder_decoder.h"
#include "common/filter.h"

Search::Search(size_t threadId)
  : scorers_(God::GetScorers(threadId)),
    filterIndices_(scorers_[0]->GetVocabSize()) {}

size_t Search::MakeFilter(const Words& srcWords, const size_t vocabSize) {
  filterIndices_ = God::GetFilter().GetFilteredVocab(srcWords, vocabSize);
  for(size_t i = 0; i < scorers_.size(); i++) {
      scorers_[i]->Filter(filterIndices_);
  }
  return filterIndices_.size();
}

History Search::Decode(const Sentence& sentence)
{
  std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

  size_t beamSize = God::Get<size_t>("beam-size");
  bool normalize = God::Get<bool>("normalize");

  History history;

  Beam prevHyps = { history.NewHypothesis() };
  history.Add(prevHyps);

  States states(scorers_.size());
  States nextStates(scorers_.size());
  Probs probs(scorers_.size());

  size_t vocabSize = scorers_[0]->GetVocabSize();

  bool filter = God::Get<std::vector<std::string>>("softmax-filter").size();
  if(filter) {
    vocabSize = MakeFilter(sentence.GetWords(), vocabSize);
  }

  for(size_t i = 0; i < scorers_.size(); i++) {
    scorers_[i]->SetSource(sentence);

    states[i].reset(scorers_[i]->NewState());
    nextStates[i].reset(scorers_[i]->NewState());

    scorers_[i]->BeginSentenceState(*states[i]);
  }

  const size_t maxLength = sentence.GetWords().size() * 3;
  do {
    for (size_t i = 0; i < scorers_.size(); i++) {
      probs[i].Resize(beamSize, vocabSize);
      scorers_[i]->Score(*states[i], probs[i], *nextStates[i]);
    }

    Beam hyps;
    BestHyps(hyps, prevHyps, probs, beamSize, history);
    history.Add(hyps, history.size() == maxLength);

    Beam survivors;
    for(auto h : hyps) {
      if(h->GetWord() != EOS)
        survivors.push_back(h);
    }
    beamSize = survivors.size();
    if(beamSize == 0)
      break;

    for(size_t i = 0; i < scorers_.size(); i++)
      scorers_[i]->AssembleBeamState(*nextStates[i], survivors, *states[i]);

    prevHyps.swap(survivors);

  } while(history.size() <= maxLength);

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::chrono::duration<double> fp_s = end - start;
  LOG(progress) << "Line " << sentence.GetLine()
    << ": Search took " << fp_s.count()
    << "s";

  CleanUp();
  return history;
}

/////////////////////////////////////////////////////////////////////
struct ProbCompare {
  ProbCompare(const float* data) : data_(data) {}

  bool operator()(const unsigned a, const unsigned b) {
    return data_[a] > data_[b];
  }

  const float* data_;
};

/////////////////////////////////////////////////////////////////////
void Search::BestHyps(Beam& bestHyps, const Beam& prevHyps,
              std::vector<mblas::Matrix>& ProbsEnsemble,
              const size_t beamSize,
              History& history) {
  using namespace mblas;

  auto& weights = God::GetScorerWeights();

  Matrix& Probs = ProbsEnsemble[0];

  Matrix Costs(Probs.Rows(), 1);
  for(int i = 0; i < prevHyps.size(); ++i)
    Costs.data()[i] = prevHyps[i]->GetCost();

  BroadcastVecColumn(weights[scorers_[0]->GetName()] * boost::phoenix::placeholders::_1 + boost::phoenix::placeholders::_2,
                     Probs, Costs);
  for(size_t i = 1; i < ProbsEnsemble.size(); ++i)
    Element(boost::phoenix::placeholders::_1 + weights[scorers_[i]->GetName()] * boost::phoenix::placeholders::_2,
            Probs, ProbsEnsemble[i]);

  std::vector<unsigned> keys(Probs.size());
  for(unsigned i = 0; i < keys.size(); ++i)
    keys[i] = i;

  std::vector<unsigned> bestKeys(beamSize);
  std::vector<float> bestCosts(beamSize);

  if(!God::Get<bool>("allow-unk")) {
    for(size_t i = 0; i < Probs.Rows(); i++)
      Probs.Set(i, UNK, std::numeric_limits<float>::lowest());
  }

  std::nth_element(keys.begin(), keys.begin() + beamSize, keys.end(),
                   ProbCompare(Probs.data()));

  for(int i = 0; i < beamSize; ++i) {
    bestKeys[i] = keys[i];
    bestCosts[i] = Probs.data()[keys[i]];
  }

  std::vector<HostVector<float>> breakDowns;
  bool doBreakdown = God::Get<bool>("n-best");
  if(doBreakdown) {
    breakDowns.push_back(bestCosts);
    for(size_t i = 1; i < ProbsEnsemble.size(); ++i) {
      HostVector<float> modelCosts(beamSize);
      auto it = boost::make_permutation_iterator(ProbsEnsemble[i].begin(), keys.begin());
      std::copy(it, it + beamSize, modelCosts.begin());
      breakDowns.push_back(modelCosts);
    }
  }

  bool filter = God::Get<std::vector<std::string>>("softmax-filter").size();
  for(size_t i = 0; i < beamSize; i++) {
    size_t wordIndex = bestKeys[i] % Probs.Cols();

    if (filter) {
      wordIndex = filterIndices_[wordIndex];
    }

    size_t hypIndex  = bestKeys[i] / Probs.Cols();
    float cost = bestCosts[i];

    HypothesisPtr hyp = history.NewHypothesis(prevHyps[hypIndex], wordIndex, hypIndex, cost);

    if(doBreakdown) {
      hyp->GetCostBreakdown().resize(ProbsEnsemble.size());
      float sum = 0;
      for(size_t j = 0; j < ProbsEnsemble.size(); ++j) {
        if(j == 0)
          hyp->GetCostBreakdown()[0] = breakDowns[0][i];
        else {
          float cost = 0;
          if(j < ProbsEnsemble.size()) {
            if(prevHyps[hypIndex]->GetCostBreakdown().size() < ProbsEnsemble.size())
              const_cast<HypothesisPtr&>(prevHyps[hypIndex])->GetCostBreakdown().resize(ProbsEnsemble.size(), 0.0);
            cost = breakDowns[j][i] + const_cast<HypothesisPtr&>(prevHyps[hypIndex])->GetCostBreakdown()[j];
          }
          sum += weights[scorers_[j]->GetName()] * cost;
          hyp->GetCostBreakdown()[j] = cost;
        }
      }
      hyp->GetCostBreakdown()[0] -= sum;
      hyp->GetCostBreakdown()[0] /= weights[scorers_[0]->GetName()];
    }
    bestHyps.push_back(hyp);
  }
}

void Search::CleanUp() {
  for(auto&& scorer : scorers_) {
    scorer->CleanUpAfterSentence();
  }
}
