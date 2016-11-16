#pragma once

#include <map>

#include "common/scorer.h"
#include "common/exception.h"
#include "common/god.h"
#include "gpu/mblas/matrix_functions.h"
#include "gpu/mblas/nth_element.h"

#include "gpu/decoder/encoder_decoder.h"

namespace GPU {

class BestHyps {
  public:
    BestHyps()
      : nthElement_(God::Get<size_t>("beam-size"), mblas::CudaStreamHandler::GetStream()),
        keys(God::Get<size_t>("beam-size")),
        Costs(God::Get<size_t>("beam-size")),
        weights_(God::GetScorerWeights())
    {}

    void DisAllowUNK(mblas::Matrix& Prob) {
      SetColumn(Prob, UNK, std::numeric_limits<float>::lowest());
    }

    void FindBests(const std::vector<size_t>& beamSizes, mblas::Matrix& Probs,
                   std::vector<float>& outCosts,
                   std::vector<unsigned>& outKeys,
                   const bool isFirst) {
      const size_t vocabSize = Probs.Cols();
      size_t batchBegin = 0;
      // std::cerr << beamSizes[0] << std::endl;
      for (size_t batchIdx = 0; batchIdx < beamSizes.size(); ++batchIdx) {
        const size_t nElements = ((isFirst) ? 1: beamSizes[batchIdx]) * vocabSize;
        // std::cerr << "N: " << nElements << std::endl;
        nthElement_.getNBestList(Probs.data() + batchBegin, nElements, beamSizes[batchIdx], outKeys, outCosts);
        for (size_t i = 0; i < beamSizes[batchIdx]; ++i) {
          outKeys[outKeys.size() - 1 - i] += batchBegin;
        }
        batchBegin += nElements;
        // std::cerr << outKeys.size() << " x " << outCosts.size() << std::endl;
      }
    }

    std::vector<SoftAlignmentPtr> GetAlignments(const std::vector<ScorerPtr>& scorers,
                                                size_t hypIndex) {
      std::vector<SoftAlignmentPtr> alignments;
      for (auto& scorer : scorers) {
        if (GPU::EncoderDecoder* encdec = dynamic_cast<GPU::EncoderDecoder*>(scorer.get())) {
          auto& attention = encdec->GetAttention();
          size_t attLength = attention.Cols();

          alignments.emplace_back(new SoftAlignment(
                attention.begin() + hypIndex * attLength,
                attention.begin() + (hypIndex + 1) * attLength));
        } else {
          UTIL_THROW2("Return Alignment is allowed only with Nematus scorer.");
        }
      }
      return alignments;
    }

    void operator()(std::vector<Beam>& beams,
          const Beam& prevHyps,
          std::vector<size_t>& beamSizes,
          const std::vector<ScorerPtr>& scorers,
          const Words& filterIndices,
          bool returnAlignment) {
      using namespace mblas;

      mblas::Matrix& Probs = static_cast<mblas::Matrix&>(scorers[0]->GetProbs());

      HostVector<float> vCosts;
      for (auto& h : prevHyps) {
        vCosts.push_back(h->GetCost());
      }
      mblas::copy(vCosts.begin(), vCosts.end(), Costs.begin());

      const bool isFirst = (vCosts[0] == 0.0f) ? true : false;

      BroadcastVecColumn(weights_[scorers[0]->GetName()] * _1 + _2, Probs, Costs);

      for (size_t i = 1; i < scorers.size(); ++i) {
        mblas::Matrix &currProbs = static_cast<mblas::Matrix&>(scorers[i]->GetProbs());

        Element(_1 + weights_[scorers[i]->GetName()] * _2, Probs, currProbs);
      }

      if (!God::Get<bool>("allow-unk")) {
        DisAllowUNK(Probs);
      }

      size_t beamSizeSum = std::accumulate(beamSizes.begin(), beamSizes.end(), 0);

      std::vector<float> bestCosts;
      std::vector<unsigned> bestKeys;

      FindBests(beamSizes, Probs, bestCosts, bestKeys, isFirst);

      std::vector<HostVector<float>> breakDowns;
      bool doBreakdown = God::Get<bool>("n-best");
      if (doBreakdown) {
          breakDowns.push_back(bestCosts);
          for (size_t i = 1; i < scorers.size(); ++i) {
            std::vector<float> modelCosts(beamSizeSum);
            mblas::Matrix &currProbs = static_cast<mblas::Matrix&>(scorers[i]->GetProbs());

            nthElement_.getValueByKey(modelCosts, currProbs.data());
            breakDowns.push_back(modelCosts);
          }
      }

      bool filter = God::Get<std::vector<std::string>>("softmax-filter").size();

      // std::cerr << "Creaing map";
      std::map<size_t, size_t> batchMap;
      size_t tmp = 0;
      for (size_t batchID = 0; batchID < beamSizes.size(); ++batchID) {
        for (size_t t = 0; t < beamSizes[batchID]; ++t) {
          // std::cerr << beamSizes.size() << " " << t << " " << batchID << " " << beamSizes[batchID] << std::endl;
          batchMap[tmp++] = batchID;
        }
      }

      // std::cerr << "MAPPING DONE." << std::endl;

      for (size_t i = 0; i < beamSizeSum; i++) {
        size_t wordIndex = bestKeys[i] % Probs.Cols();
        if (filter) {
          wordIndex = filterIndices[wordIndex];
        }

        size_t hypIndex  = bestKeys[i] / Probs.Cols();
        float cost = bestCosts[i];

        HypothesisPtr hyp;
        if (returnAlignment) {
          hyp.reset(new Hypothesis(prevHyps[hypIndex], wordIndex, hypIndex, cost,
                                   GetAlignments(scorers, hypIndex)));
        } else {
          hyp.reset(new Hypothesis(prevHyps[hypIndex], wordIndex, hypIndex, cost));
        }

        if(doBreakdown) {
          hyp->GetCostBreakdown().resize(scorers.size());
          float sum = 0;
          for (size_t j = 0; j < scorers.size(); ++j) {
            if (j == 0)
              hyp->GetCostBreakdown()[0] = breakDowns[0][i];
            else {
              float cost = 0;
              if (j < scorers.size()) {
                  if (prevHyps[hypIndex]->GetCostBreakdown().size() < scorers.size())
                    const_cast<HypothesisPtr&>(prevHyps[hypIndex])->GetCostBreakdown().resize(scorers.size(), 0.0f);
                  cost = breakDowns[j][i] + const_cast<HypothesisPtr&>(prevHyps[hypIndex])->GetCostBreakdown()[j];
              }
              sum += weights_[scorers[j]->GetName()] * cost;
              hyp->GetCostBreakdown()[j] = cost;
            }
          }
          hyp->GetCostBreakdown()[0] -= sum;
          hyp->GetCostBreakdown()[0] /= weights_[scorers[0]->GetName()];
        }

      // std::cerr << i << ": " << batchMap[i] << std::endl;
      beams[batchMap[i]].push_back(hyp);
      }
    }

  private:
    NthElement nthElement_;
    DeviceVector<unsigned> keys;
    DeviceVector<float> Costs;
    std::map<std::string, float>& weights_;
};

}
