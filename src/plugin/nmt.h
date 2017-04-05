#include <iostream>
#include <vector>
#include <set>

#include "nbest.h"
#include "common/scorer.h"



typedef std::vector<int> Batch;
typedef std::vector<Batch> Batches;
typedef std::vector<float> Scores;

namespace amunmt {

class God;

class NMT {
  public:
    NMT(std::vector<ScorerPtr>& scorers);

    static void InitGod(const std::string& configFilePath);

    static std::vector<ScorerPtr> NewScorers();

    static size_t GetTotalThreads();

    static size_t GetDevices(size_t = 1);

    std::vector<ScorerPtr>& GetScorers() {
      return scorers_;
    }

    void SetDevice();
    size_t GetDevice();

    void SetDebug(bool debug) {
      debug_ = debug;
    }

    void ClearStates();

    States NewStates() const;

    States CalcSourceContext(const std::vector<std::string>& s);

    size_t TargetVocab(const std::string& str);

    void BatchSteps(const Batches& batches,
                    Scores& probs,
                    Scores& unks,
                    std::vector<States>& inputStates);

    void OnePhrase(
      const std::vector<std::string>& phrase,
      const States& inputStates,
      float& prob,
      size_t& unks,
      States& outputStates);


    std::vector<double> RescoreNBestList(
        const std::vector<std::string>& nbest,
        const size_t maxBatchSize=64);

  private:
    bool debug_;
    static std::shared_ptr<God> god_;

    std::vector<ScorerPtr> scorers_;
    Words filterIndices_;

    std::shared_ptr<States> states_;
    bool firstWord_;

    std::vector<size_t> filteredId_;
};

}
