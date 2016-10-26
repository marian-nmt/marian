#pragma once

#include "common/scorer.h"
#include "gpu/mblas/matrix.h"
#include "gpu/decoder/nth_element.h"

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

namespace GPU {

class BestHyps {
  public:
    BestHyps()
      : nthElement_(God::Get<size_t>("beam-size"), mblas::Matrix::GetStream()),
        keys(God::Get<size_t>("beam-size")),
        Costs(God::Get<size_t>("beam-size")),
        weights_(God::GetScorerWeights())
    {}

    void DisAllowUNK(mblas::Matrix& Prob) {
      SetColumn(Prob, UNK, std::numeric_limits<float>::lowest());
    }

    void FindBests(size_t beamSize, mblas::Matrix& Probs,
                   std::vector<float>& outCosts,
                   std::vector<unsigned>& outKeys) {
      size_t nElements = Probs.Cols() * Probs.Rows();
      nthElement_.getNBestList(Probs.data(), nElements, beamSize, outKeys, outCosts);
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

    void operator()(Beam& bestHyps,
          const Beam& prevHyps,
          const size_t beamSize,
          const std::vector<ScorerPtr>& scorers,
          const Words& filterIndices,
          bool returnAlignment) {
      using namespace mblas;

      mblas::Matrix& Probs = static_cast<mblas::Matrix&>(scorers[0]->GetProbs());

      HostVector<float> vCosts;
      for (auto& h : prevHyps) {
        vCosts.push_back(h->GetCost());
      }
      thrust::copy(thrust::cuda::par.on(Matrix::GetStream()),
                   vCosts.begin(), vCosts.end(), Costs.begin());

      BroadcastVecColumn(weights_[scorers[0]->GetName()] * _1 + _2, Probs, Costs);

      for (size_t i = 1; i < scorers.size(); ++i) {
        mblas::Matrix &currProbs = static_cast<mblas::Matrix&>(scorers[i]->GetProbs());

        Element(_1 + weights_[scorers[i]->GetName()] * _2, Probs, currProbs);
      }

      if (!God::Get<bool>("allow-unk")) {
        DisAllowUNK(Probs);
      }

      std::vector<float> bestCosts(beamSize);
      std::vector<unsigned> bestKeys(beamSize);

      FindBests(beamSize, Probs, bestCosts, bestKeys);

      std::vector<HostVector<float>> breakDowns;
      bool doBreakdown = God::Get<bool>("n-best");
      if (doBreakdown) {
          breakDowns.push_back(bestCosts);
          for (size_t i = 1; i < scorers.size(); ++i) {
            HostVector<float> modelCosts(beamSize);
            mblas::Matrix &currProbs = static_cast<mblas::Matrix&>(scorers[i]->GetProbs());

            auto it = thrust::make_permutation_iterator(currProbs.begin(), keys.begin());
            algo::copy(thrust::cuda::par.on(Matrix::GetStream()),
                        it, it + beamSize, modelCosts.begin());
            breakDowns.push_back(modelCosts);
          }
      }

      bool filter = God::Get<std::vector<std::string>>("softmax-filter").size();

      for (size_t i = 0; i < beamSize; i++) {
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
      bestHyps.push_back(hyp);
      }
    }

  private:
    NthElement nthElement_;
    thrust::device_vector<unsigned> keys;
    thrust::device_vector<float> Costs;
    std::map<std::string, float>& weights_;
};

}
