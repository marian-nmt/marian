#pragma once

#include <vector>
#include <boost/iterator/permutation_iterator.hpp>

#include "common/scorer.h"
#include "common/god.h"
#include "common/exception.h"
#include "cpu/mblas/matrix.h"
#include "cpu/decoder/encoder_decoder.h"

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
    BestHyps(const God &god)
      : BestHypsBase(god,
          !god.Get<bool>("allow-unk"),
          god.Get<std::vector<std::string>>("softmax-filter").size(),
          god.Get<bool>("return-alignment") || god.Get<bool>("return-soft-alignment") || god.Get<bool>("return-nematus-alignment"),
          god.GetScorerWeights())
    {}

    void CalcBeam(
        const Beam& prevHyps,
        const std::vector<ScorerPtr>& scorers,
        const Words& filterIndices,
        std::vector<Beam>& beams,
        std::vector<uint>& beamSizes)
    {
      using namespace mblas;

      mblas::ArrayMatrix& Probs = static_cast<mblas::ArrayMatrix&>(scorers[0]->GetProbs());

      mblas::ArrayMatrix Costs(Probs.rows(), 1);
      for (size_t i = 0; i < prevHyps.size(); ++i) {
        Costs.data()[i] = prevHyps[i]->GetCost();
      }

      Probs *= weights_.at(scorers[0]->GetName());
      AddBiasVector<byColumn>(Probs, Costs);

      for (size_t i = 1; i < scorers.size(); ++i) {
        mblas::ArrayMatrix &currProb = static_cast<mblas::ArrayMatrix&>(scorers[i]->GetProbs());

        Probs += weights_.at(scorers[i]->GetName()) * currProb;
      }

      size_t size = Probs.rows() * Probs.columns(); // Probs.size();
      std::vector<size_t> keys(size);
      for (size_t i = 0; i < keys.size(); ++i) {
        keys[i] = i;
      }

      size_t beamSize = beamSizes[0];

      std::vector<size_t> bestKeys(beamSize);
      std::vector<float> bestCosts(beamSize);

      if (forbidUNK_) {
        blaze::column(Probs, UNK_ID) = std::numeric_limits<float>::lowest();
      }

      std::nth_element(keys.begin(), keys.begin() + beamSize, keys.end(),
                       ProbCompare(Probs.data()));

      for (size_t i = 0; i < beamSize; ++i) {
        bestKeys[i] = keys[i];
        bestCosts[i] = Probs.data()[keys[i]];
      }

      std::vector<std::vector<float>> breakDowns;
      if (god_.ReturnNBestList()) {
        breakDowns.push_back(bestCosts);
        for (auto& scorer : scorers) {
          std::vector<float> modelCosts(beamSize);
          mblas::ArrayMatrix &currProb = static_cast<mblas::ArrayMatrix&>(scorer->GetProbs());

          auto it = boost::make_permutation_iterator(currProb.begin(), keys.begin());
          std::copy(it, it + beamSize, modelCosts.begin());
          breakDowns.push_back(modelCosts);
        }
      }

      for (size_t i = 0; i < beamSize; i++) {
        size_t wordIndex = bestKeys[i] % Probs.columns();

        if (isInputFiltered_) {
          wordIndex = filterIndices[wordIndex];
        }

        size_t hypIndex  = bestKeys[i] / Probs.columns();
        float cost = bestCosts[i];

        HypothesisPtr hyp;
        if (returnAttentionWeights_) {
          std::vector<SoftAlignmentPtr> alignments;
          for (auto& scorer : scorers) {
            if (CPU::CPUEncoderDecoderBase* encdec = dynamic_cast<CPU::CPUEncoderDecoderBase*>(scorer.get())) {
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

        if (god_.ReturnNBestList()) {
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
              sum += weights_.at(scorers[j]->GetName()) * cost;
              hyp->GetCostBreakdown()[j] = cost;
            }
          }
          hyp->GetCostBreakdown()[0] -= sum;
          hyp->GetCostBreakdown()[0] /= weights_.at(scorers[0]->GetName());
        }
        beams[0].push_back(hyp);
      }
    }
};

}  // namespace CPU
}  // namespace amunmt
