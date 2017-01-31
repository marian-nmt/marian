#pragma once

#include <vector>
#include <boost/iterator/permutation_iterator.hpp>

#include "common/scorer.h"
#include "common/god.h"
#include "common/exception.h"
#include "cpu/mblas/matrix.h"

namespace amunmt {
namespace CPU {

struct ProbCompare {
  ProbCompare(const float* data) : data_(data) {}

  bool operator()(const unsigned a, const unsigned b) {
    return data_[a] > data_[b];
  }

  const float* data_;
};

class BestHyps : public BestHypsBase
{
public:
  void CalcBeam(
      const God &god,
      const Beam& prevHyps,
      const std::vector<ScorerPtr>& scorers,
      const Words& filterIndices,
      bool returnAlignment,
      std::vector<Beam>& beams,
      std::vector<size_t>& beamSizes
      )
  {
    using namespace mblas;

    auto& weights = god.GetScorerWeights();

    mblas::ArrayMatrix& Probs = static_cast<mblas::ArrayMatrix&>(scorers[0]->GetProbs());

    mblas::ArrayMatrix Costs(Probs.rows(), 1);
    for (size_t i = 0; i < prevHyps.size(); ++i) {
      Costs.data()[i] = prevHyps[i]->GetCost();
    }

    Probs *= weights.at(scorers[0]->GetName());
    AddBiasVector<byColumn>(Probs, Costs);

    for (size_t i = 1; i < scorers.size(); ++i) {
      mblas::ArrayMatrix &currProb = static_cast<mblas::ArrayMatrix&>(scorers[i]->GetProbs());

      Probs += weights.at(scorers[i]->GetName()) * currProb;
    }

    size_t size = Probs.rows() * Probs.columns(); // Probs.size();
    std::vector<size_t> keys(size);
    for (size_t i = 0; i < keys.size(); ++i) {
      keys[i] = i;
    }

    size_t beamSize = beamSizes[0];

    std::vector<size_t> bestKeys(beamSize);
    std::vector<float> bestCosts(beamSize);

    if (!god.Get<bool>("allow-unk")) {
      blaze::column(Probs, UNK) = std::numeric_limits<float>::lowest();
    }

    std::nth_element(keys.begin(), keys.begin() + beamSize, keys.end(),
                 ProbCompare(Probs.data()));

    for (size_t i = 0; i < beamSize; ++i) {
      bestKeys[i] = keys[i];
      bestCosts[i] = Probs.data()[keys[i]];
    }

    std::vector<std::vector<float>> breakDowns;
    bool doBreakdown = god.Get<bool>("n-best");
    if (doBreakdown) {
      breakDowns.push_back(bestCosts);
      for (auto& scorer : scorers) {
        std::vector<float> modelCosts(beamSize);
        mblas::ArrayMatrix &currProb = static_cast<mblas::ArrayMatrix&>(scorer->GetProbs());

        auto it = boost::make_permutation_iterator(currProb.begin(), keys.begin());
        std::copy(it, it + beamSize, modelCosts.begin());
        breakDowns.push_back(modelCosts);
      }
    }

    bool filter = god.Get<std::vector<std::string>>("softmax-filter").size();

    for (size_t i = 0; i < beamSize; i++) {
      size_t wordIndex = bestKeys[i] % Probs.columns();

      if (filter) {
        wordIndex = filterIndices[wordIndex];
      }

      size_t hypIndex  = bestKeys[i] / Probs.columns();
      float cost = bestCosts[i];

      HypothesisPtr hyp;
      if (returnAlignment) {
        std::vector<SoftAlignmentPtr> alignments;
        for (auto& scorer : scorers) {
          if (CPU::EncoderDecoder* encdec = dynamic_cast<CPU::EncoderDecoder*>(scorer.get())) {
            auto& attention = encdec->GetAttention();
            alignments.emplace_back(new SoftAlignment(attention.begin(hypIndex),
                                                      attention.end(hypIndex)));
          } else {
            amunmt_UTIL_THROW2("Return Alignment is allowed only with Nematus scorer.");
          }
        }

        hyp.reset(new Hypothesis(prevHyps[hypIndex], wordIndex, hypIndex, cost, alignments));
      } else {
        hyp.reset(new Hypothesis(prevHyps[hypIndex], wordIndex, hypIndex, cost));
      }

      if (doBreakdown) {
        hyp->GetCostBreakdown().resize(scorers.size());
        float sum = 0;
        for(size_t j = 0; j < scorers.size(); ++j) {
          if (j == 0) {
            hyp->GetCostBreakdown()[0] = breakDowns[0][i];
          } else {
            float cost = 0;
            if (j < scorers.size()) {
              if (prevHyps[hypIndex]->GetCostBreakdown().size() < scorers.size())
                const_cast<HypothesisPtr&>(prevHyps[hypIndex])->GetCostBreakdown().resize(scorers.size(), 0.0);
              cost = breakDowns[j][i] + const_cast<HypothesisPtr&>(prevHyps[hypIndex])->GetCostBreakdown()[j];
            }
            sum += weights.at(scorers[j]->GetName()) * cost;
            hyp->GetCostBreakdown()[j] = cost;
          }
        }
        hyp->GetCostBreakdown()[0] -= sum;
        hyp->GetCostBreakdown()[0] /= weights.at(scorers[0]->GetName());
      }
      beams[0].push_back(hyp);
    }
  }
};

} // namespace
}
