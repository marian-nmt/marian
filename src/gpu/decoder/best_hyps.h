#pragma once

#include "common/scorer.h"
#include "gpu/mblas/matrix.h"

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

namespace GPU {

struct ProbCompare {
  ProbCompare(const float* data) : data_(data) {}

  __host__ __device__
  bool operator()(const unsigned a, const unsigned b) {
    return data_[a] > data_[b];
  }

  const float* data_;
};

class BestHyps {
  public:
    void operator()(Beam& bestHyps,
          const Beam& prevHyps,
          const size_t beamSize,
          const std::vector<ScorerPtr>& scorers,
          const Words& filterIndices,
          bool returnAlignment) {
        using namespace mblas;

        auto& weights = God::GetScorerWeights();

        mblas::Matrix& Probs = static_cast<mblas::Matrix&>(scorers[0]->GetProbs());

        Costs.reserve(Probs.Rows());
        HostVector<float> vCosts;
        for (auto& h : prevHyps) {
          vCosts.push_back(h->GetCost());
        }
        thrust::copy(thrust::cuda::par.on(Matrix::GetStream()),
                  vCosts.begin(), vCosts.end(), Costs.begin());

        BroadcastVecColumn(weights[scorers[0]->GetName()] * _1 + _2,
                          Probs, Costs);
        for (size_t i = 1; i < scorers.size(); ++i) {
          mblas::Matrix &currProbs = static_cast<mblas::Matrix&>(scorers[i]->GetProbs());

          Element(_1 + weights[scorers[i]->GetName()] * _2,
                  Probs, currProbs);
        }

        keys.resize(Probs.size());
        thrust::host_vector<unsigned> bestKeys(beamSize);
        thrust::host_vector<float> bestCosts(beamSize);

        // @TODO: make this more efficient
        if (!God::Get<bool>("allow-unk")) {
          for (size_t i = 0; i < Probs.Rows(); i++)
            Probs.Set(i, UNK, std::numeric_limits<float>::lowest());
        }

        // @TODO: Here we need to have a partial sort
        if (beamSize < 10) {
          for (size_t i = 0; i < beamSize; ++i) {
            DeviceVector<float>::iterator iter =
            thrust::max_element(thrust::cuda::par.on(Matrix::GetStream()),
                                Probs.begin(), Probs.end());
            bestKeys[i] = iter - Probs.begin();
            bestCosts[i] = *iter;
            *iter = std::numeric_limits<float>::lowest();
          }
          algo::copy(thrust::cuda::par.on(Matrix::GetStream()),
                    bestKeys.begin(), bestKeys.end(), keys.begin());
        }
        else {
            // these two function do not have equivalents in
            // in the standard library or boost, keeping thrust
            // namespace for now
            thrust::sequence(thrust::cuda::par.on(Matrix::GetStream()),
                            keys.begin(), keys.end());
            thrust::sort_by_key(thrust::cuda::par.on(Matrix::GetStream()), Probs.begin(), Probs.end(),
                                keys.begin(), algo::greater<float>());

            thrust::copy_n(thrust::cuda::par.on(Matrix::GetStream()),
                        keys.begin(), beamSize, bestKeys.begin());
            thrust::copy_n(thrust::cuda::par.on(Matrix::GetStream()),
                        Probs.begin(), beamSize, bestCosts.begin());
        }


        std::vector<HostVector<float>> breakDowns;
        bool doBreakdown = God::Get<bool>("n-best");
        if (doBreakdown) {
            breakDowns.push_back(bestCosts);
            for (size_t i = 1; i < scorers.size(); ++i) {
                HostVector<float> modelCosts(beamSize);
                mblas::Matrix &currProbs = static_cast<mblas::Matrix&>(scorers[i]->GetProbs());

                auto it = iteralgo::make_permutation_iterator(currProbs.begin(), keys.begin());
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
            std::vector<SoftAlignmentPtr> alignments;
            for (auto& scorer : scorers) {
              if (GPU::EncoderDecoder* encdec = dynamic_cast<GPU::EncoderDecoder*>(scorer.get())) {
                auto& attention = encdec->GetAttention();
                size_t attLength = attention.Cols();

                alignments.emplace_back(new SoftAlignment(attention.begin() + hypIndex * attLength,
                                                          attention.begin() + (hypIndex + 1) * attLength));
              } else {
                UTIL_THROW2("Return Alignment is allowed only with Nematus scorer.");
              }
            }
            hyp.reset(new Hypothesis(prevHyps[hypIndex], wordIndex, hypIndex, cost, alignments));
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
                  if(prevHyps[hypIndex]->GetCostBreakdown().size() < scorers.size())
                  const_cast<HypothesisPtr&>(prevHyps[hypIndex])->GetCostBreakdown().resize(scorers.size(), 0.0);
                  cost = breakDowns[j][i] + const_cast<HypothesisPtr&>(prevHyps[hypIndex])->GetCostBreakdown()[j];
              }
              sum += weights[scorers[j]->GetName()] * cost;
              hyp->GetCostBreakdown()[j] = cost;
              }
          }
          hyp->GetCostBreakdown()[0] -= sum;
          hyp->GetCostBreakdown()[0] /= weights[scorers[0]->GetName()];
        }
        bestHyps.push_back(hyp);
      }
    }

  private:
    mutable thrust::device_vector<unsigned> keys;
    mutable thrust::device_vector<float> Costs;

};

}
