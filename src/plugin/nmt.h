#include <iostream>
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
    NeuralExtention(const std::vector<size_t>& phrase, float score, const std::vector<size_t>& coverage,
                    size_t prevIndex)
      : phrase_(phrase), score_(score), coverage_(coverage), prevIndex_(prevIndex)
    {}

  protected:
    std::vector<size_t> phrase_;
    float score_;
    std::vector<size_t> coverage_;
    size_t prevIndex_;
};

class NMT {
  public:
    NMT();
    NMT(std::vector<ScorerPtr>& scorers);
    virtual ~NMT();

    static void InitGod(const std::string& configFilePath);
    static void Clean();

    static std::vector<ScorerPtr> NewScorers();

    static size_t GetTotalThreads();
    static size_t GetBatchSize();

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

    std::vector<double> RescoreNBestList(
        const std::vector<std::string>& nbest,
        const size_t maxBatchSize=64);

    std::vector<NeuralExtention> GetNeuralExtentions(std::vector<States>& inputStates);

  private:
    bool debug_;
    static std::shared_ptr<God> god_;

    std::vector<ScorerPtr> scorers_;
    BestHypsBasePtr bestHyps_;
    Words filterIndices_;

    bool firstWord_;

    std::vector<size_t> filteredId_;
};

}
