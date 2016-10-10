#include <boost/iterator/permutation_iterator.hpp>
#include "cpu/mblas/matrix.h"
#include "cpu/mblas/simd_math_prims.h"
#include "common/god.h"
#include "common/hypothesis.h"
#include "common/history.h"
#include "common/soft_alignment.h"
#include "common/exception.h"
#include "gpu/types-gpu.h"

#include "blaze/Math.h"

#include "cpu/decoder/encoder_decoder.h"

using namespace std;

namespace CPU {

namespace mblas {

/////////////////////////////////////////////////////////////////////
struct ProbCompare {
  ProbCompare(const float* data) : data_(data) {}

  bool operator()(const unsigned a, const unsigned b) {
    return data_[a] > data_[b];
  }

  const float* data_;
};

/////////////////////////////////////////////////////////////////////
void ArrayMatrix::BestHyps(Beam& bestHyps,
    const Beam& prevHyps,
		BaseMatrices& ProbsEnsemble,
		const size_t beamSize,
		History& history,
		const std::vector<ScorerPtr> &scorers,
		const Words &filterIndices,
    bool returnAlignment) const
{
  using namespace mblas;

  auto& weights = God::GetScorerWeights();

  mblas::ArrayMatrix& Probs = static_cast<mblas::ArrayMatrix&>(*ProbsEnsemble[0]);

  mblas::ArrayMatrix Costs(Probs.rows(), 1);
  for (size_t i = 0; i < prevHyps.size(); ++i) {
    Costs.data()[i] = prevHyps[i]->GetCost();
  }

  Probs *= weights[scorers[0]->GetName()];
  AddBiasVector<byColumn>(Probs, Costs);

  for (size_t i = 1; i < ProbsEnsemble.size(); ++i) {
	  mblas::ArrayMatrix &currProb = static_cast<mblas::ArrayMatrix&>(*ProbsEnsemble[i]);

	  Probs += weights[scorers[i]->GetName()] * currProb;
  }

  size_t size = Probs.rows() * Probs.columns(); // Probs.size();
  std::vector<size_t> keys(size);
  for (size_t i = 0; i < keys.size(); ++i) {
    keys[i] = i;
  }

  std::vector<size_t> bestKeys(beamSize);
  std::vector<float> bestCosts(beamSize);

  if (!God::Get<bool>("allow-unk")) {
    blaze::column(Probs, UNK) = std::numeric_limits<float>::lowest();
  }

  std::nth_element(keys.begin(), keys.begin() + beamSize, keys.end(),
		           ProbCompare(Probs.data()));

  for (int i = 0; i < beamSize; ++i) {
    bestKeys[i] = keys[i];
    bestCosts[i] = Probs.data()[keys[i]];
  }

  std::vector<HostVector<float>> breakDowns;
  bool doBreakdown = God::Get<bool>("n-best");
  if (doBreakdown) {
    breakDowns.push_back(bestCosts);
    for (size_t i = 1; i < ProbsEnsemble.size(); ++i) {
      HostVector<float> modelCosts(beamSize);
      mblas::ArrayMatrix &currProb = static_cast<mblas::ArrayMatrix&>(*ProbsEnsemble[i]);

      auto it = boost::make_permutation_iterator(currProb.begin(), keys.begin());
      std::copy(it, it + beamSize, modelCosts.begin());
      breakDowns.push_back(modelCosts);
    }
  }

  bool filter = God::Get<std::vector<std::string>>("softmax-filter").size();

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
          size_t attLength = attention.columns();
          alignments.emplace_back(new SoftAlignment(attention.begin(hypIndex), attention.end(hypIndex)));
        } else {
          UTIL_THROW2("Return Alignment is allowed only with Nematus scorer.");
        }
      }

      hyp.reset(new Hypothesis(prevHyps[hypIndex], wordIndex, hypIndex, cost, alignments));
    } else {
      hyp.reset(new Hypothesis(prevHyps[hypIndex], wordIndex, hypIndex, cost));
    }

    if (doBreakdown) {
      hyp->GetCostBreakdown().resize(ProbsEnsemble.size());
      float sum = 0;
      for(size_t j = 0; j < ProbsEnsemble.size(); ++j) {
      if (j == 0) {
        hyp->GetCostBreakdown()[0] = breakDowns[0][i];
      } else {
        float cost = 0;
        if (j < ProbsEnsemble.size()) {
          if (prevHyps[hypIndex]->GetCostBreakdown().size() < ProbsEnsemble.size())
            const_cast<HypothesisPtr&>(prevHyps[hypIndex])->GetCostBreakdown().resize(ProbsEnsemble.size(), 0.0);
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


}
}

