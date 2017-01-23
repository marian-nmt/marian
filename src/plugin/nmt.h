#pragma once

#include <iostream>
#include <algorithm>
#include <vector>
#include <set>
#include <memory>

#include "common/god.h"
#include "common/scorer.h"
#include "common/sentence.h"
#include "gpu/mblas/matrix.h"
#include "gpu/decoder/encoder_decoder.h"
#include "neural_phrase.h"
#include "hypo_info.h"

class Vocab;

class StateInfo;
class NeuralPhrase;
typedef std::shared_ptr<StateInfo> StateInfoPtr;

typedef std::vector<size_t> Batch;
typedef std::vector<Batch> Batches;
typedef std::vector<StateInfoPtr> StateInfos;
typedef std::vector<float> Scores;
typedef std::vector<size_t> LastWords;

class MosesPlugin {
  public:
    MosesPlugin();
		~MosesPlugin();
		
    static size_t GetDevices(size_t = 1);
    void SetDevice();
    size_t GetDevice();

    void SetDebug(bool debug) {
      debug_ = debug;
    }

    void initGod(const std::string& configPath);

    States SetSource(const std::vector<size_t>& words);

    void FilterTargetVocab(const std::set<std::string>& filter, size_t topN);

    size_t TargetVocab(const std::string& str);
    size_t SourceVocab(const std::string& str);

    // void BatchSteps(const Batches& batches, LastWords& lastWords,
                    // Scores& probs, Scores& unks, StateInfos& stateInfos,
                    // bool firstWord);

    // void OnePhrase(
      // const std::vector<std::string>& phrase,
      // const std::string& lastWord,
      // bool firstWord,
      // StateInfoPtr inputState,
      // float& prob, size_t& unks,
      // StateInfoPtr& outputState);

    // void MakeStep(
      // const std::vector<std::string>& nextWords,
      // const std::vector<std::string>& lastWords,
      // std::vector<StateInfoPtr>& inputStates,
      // std::vector<double>& logProbs,
      // std::vector<StateInfoPtr>& nextStates,
      // std::vector<bool>& unks);

    // void ClearStates();

    // std::vector<double> RescoreNBestList(
        // const std::vector<std::string>& nbest,
        // const size_t maxBatchSize=64);
    void GeneratePhrases(const States& states, size_t lastWord, size_t numPhrases,
                         std::vector<NeuralPhrase>& phrases);

    States GenerateStates(const States& parentStates, size_t lastWord, std::vector<size_t>& phrase);
    void Rescore(std::vector<HypoInfo> &hypos);

  private:
    bool debug_;

    God god_;
    
    std::vector<ScorerPtr> scorers_;
    Words filterIndices_;
    BestHypsBasePtr bestHyps_;
    Sentences sentences_;

    std::shared_ptr<States> states_;
    bool firstWord_;

    std::vector<size_t> filteredId_;
};
