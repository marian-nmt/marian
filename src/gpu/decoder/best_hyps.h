#pragma once

#include <map>
#include <numeric>

#include "common/scorer.h"
#include "common/exception.h"
#include "common/god.h"
#include "gpu/mblas/matrix_functions.h"
#include "gpu/mblas/nth_element.h"

#include "gpu/decoder/encoder_decoder.h"

namespace amunmt {
namespace GPU {

class BestHyps : public BestHypsBase
{
  public:
    BestHyps(const BestHyps &copy) = delete;
    BestHyps(const God &god)
    : nthElement_(god.Get<size_t>("beam-size"), god.Get<size_t>("mini-batch"),
                  mblas::CudaStreamHandler::GetStream()),
      keys(god.Get<size_t>("beam-size") * god.Get<size_t>("mini-batch")),
      Costs(god.Get<size_t>("beam-size") * god.Get<size_t>("mini-batch")),
      weights_(god.GetScorerWeights())
    {
      //std::cerr << "BestHyps::BestHyps" << std::endl;
    }

    void DisAllowUNK(mblas::Matrix& Prob) {
      SetColumn(Prob, UNK, std::numeric_limits<float>::lowest());
    }

    void FindBests(const std::vector<size_t>& beamSizes, mblas::Matrix& Probs,
                   std::vector<float>& outCosts,
                   std::vector<unsigned>& outKeys,
                   const bool isFirst) {
      nthElement_.getNBestList(beamSizes, Probs, outCosts, outKeys, isFirst);
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
          amunmt_UTIL_THROW2("Return Alignment is allowed only with Nematus scorer.");
        }
      }
      return alignments;
    }

    void CalcBeam(const God &god,
          const Beam& prevHyps,
          const std::vector<ScorerPtr>& scorers,
          const Words& filterIndices,
          bool returnAlignment,
          std::vector<Beam>& beams,
          std::vector<size_t>& beamSizes
          )
    {
      using namespace mblas;

      mblas::Matrix& Probs = static_cast<mblas::Matrix&>(scorers[0]->GetProbs());

      HostVector<float> vCosts;
      for (auto& h : prevHyps) {
        vCosts.push_back(h->GetCost());
      }
      mblas::copy(vCosts.begin(), vCosts.end(), Costs.begin());

      const bool isFirst = (vCosts[0] == 0.0f) ? true : false;

      BroadcastVecColumn(weights_.at(scorers[0]->GetName()) * _1 + _2, Probs, Costs);

      for (size_t i = 1; i < scorers.size(); ++i) {
        mblas::Matrix &currProbs = static_cast<mblas::Matrix&>(scorers[i]->GetProbs());

        Element(_1 + weights_.at(scorers[i]->GetName()) * _2, Probs, currProbs);
      }

      if (!god.Get<bool>("allow-unk")) {
        DisAllowUNK(Probs);
      }

      size_t beamSizeSum = std::accumulate(beamSizes.begin(), beamSizes.end(), 0);

      std::vector<float> bestCosts;
      std::vector<unsigned> bestKeys;

      FindBests(beamSizes, Probs, bestCosts, bestKeys, isFirst);

      std::vector<HostVector<float>> breakDowns;
      bool doBreakdown = god.Get<bool>("n-best");
      if (doBreakdown) {
          breakDowns.push_back(bestCosts);
          for (size_t i = 1; i < scorers.size(); ++i) {
            std::vector<float> modelCosts(beamSizeSum);
            mblas::Matrix &currProbs = static_cast<mblas::Matrix&>(scorers[i]->GetProbs());

            nthElement_.getValueByKey(modelCosts, currProbs.data());
            breakDowns.push_back(modelCosts);
          }
      }

      bool filter = god.Get<std::vector<std::string>>("softmax-filter").size();

      std::map<size_t, size_t> batchMap;
      size_t tmp = 0;
      for (size_t batchID = 0; batchID < beamSizes.size(); ++batchID) {
        for (size_t t = 0; t < beamSizes[batchID]; ++t) {
          batchMap[tmp++] = batchID;
        }
      }

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
              sum += weights_.at(scorers[j]->GetName()) * cost;
              hyp->GetCostBreakdown()[j] = cost;
            }
          }
          hyp->GetCostBreakdown()[0] -= sum;
          hyp->GetCostBreakdown()[0] /= weights_.at(scorers[0]->GetName());
        }

        beams[batchMap[i]].push_back(hyp);
      }
    }

  private:
    NthElement nthElement_;
    DeviceVector<unsigned> keys;
    DeviceVector<float> Costs;
    const std::map<std::string, float>& weights_;
};

}
}

