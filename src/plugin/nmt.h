#pragma once
#include <iostream>
#include <iomanip>
#include <vector>
#include <set>

#include "nbest.h"
#include "common/scorer.h"
#include "common/base_best_hyps.h"



typedef std::vector<int> Batch;
typedef std::vector<Batch> Batches;
typedef std::vector<float> Scores;

namespace amunmt {

class God;

class NeuralExtention {
  public:
    friend std::ostream& operator<<(std::ostream& stream, const NeuralExtention& ne) {
      stream << "cost: " <<std::setw(6) << ne.score_ << ";  phrase: ";
      for (auto word : ne.phrase_)  {
        stream << word << " ";
      }

      stream << " Align: ";
      for (auto word : ne.coverage_)  {
        stream << word << " ";
      }

      stream << " PREV: " << ne.prevIndex_;
      return stream;
    }

    NeuralExtention(const std::vector<std::string>& phrase, float score, const std::vector<size_t>& coverage,
                    size_t prevIndex)
      : phrase_(phrase), score_(score), coverage_(coverage), prevIndex_(prevIndex)
    {}

  public:
    std::vector<std::string> phrase_;
    float score_;
    std::vector<size_t> coverage_;
    size_t prevIndex_;
};

class NMT {
  public:
    NMT();
    virtual ~NMT();

    static void InitGod(const std::string& configFilePath);
    static void Clean();

    static size_t GetTotalThreads();
    static size_t GetBatchSize();

    States CalcSourceContext(const std::vector<std::string>& s);

    size_t TargetVocab(const std::string& str);

    void RescorePhrases(
        const std::vector<std::vector<std::string>>& phrases,
        std::vector<States>& inputStates,
        Scores& probs);

    std::vector<float> RescoreNBestList(const std::vector<std::string>& nbest);

    std::vector<NeuralExtention> ExtendHyps( const std::vector<States>& inputStates);

  protected:
    std::vector<float> Rescore(NBest& nBest, bool returnFinalStates=false);
    void SetDevice();

    States NewStates() const;
    Beam GetSurvivors(RescoreBatch& rescoreBatch, size_t step);
    States JoinStates(const std::vector<States*>& states);
    void SaveFinalStates(const States& inStates, size_t step, RescoreBatch& rescoreBatch);

    static std::shared_ptr<God> god_;
    std::vector<ScorerPtr> scorers_;
    BestHypsBasePtr bestHyps_;
};

}  // namespace amunnmt
