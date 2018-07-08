#pragma once

#include <vector>

#include "common/config.h"
#include "common/utils.h"
#include "data/vocab.h"
#include "translator/history.h"
#include "translator/hypothesis.h"

namespace marian {

typedef std::vector<float> SoftAlignment;
typedef std::pair<size_t, size_t> HardAlignment;

class OutputPrinter {
public:
  OutputPrinter(Ptr<Config> options, Ptr<Vocab> vocab)
      : vocab_(vocab),
        reverse_(options->get<bool>("right-left")),
        nbest_(options->get<bool>("n-best", false)
                   ? options->get<size_t>("beam-size")
                   : 0),
        alignment_(options->get<float>("alignment", 0.f)) {}

  template <class OStream>
  void print(Ptr<History> history, OStream& best1, OStream& bestn) {
    const auto& nbl = history->NBest(nbest_);

    for(size_t i = 0; i < nbl.size(); ++i) {
      const auto& result = nbl[i];
      const auto& words = std::get<0>(result);
      const auto& hypo = std::get<1>(result);

      std::string translation = Join((*vocab_)(words), " ", reverse_);
      bestn << history->GetLineNum() << " ||| " << translation;

      if(alignment_ > 0.f) {
        auto align = getAlignment(hypo, alignment_);
        bestn << getAlignmentString(align);
      }

      bestn << " |||";

      if(hypo->GetCostBreakdown().empty()) {
        bestn << " F0=" << hypo->GetCost();
      } else {
        for(size_t j = 0; j < hypo->GetCostBreakdown().size(); ++j) {
          bestn << " F" << j << "= " << hypo->GetCostBreakdown()[j];
        }
      }

      float realCost = std::get<2>(result);
      bestn << " ||| " << realCost;

      if(i < nbl.size() - 1)
        bestn << std::endl;
      else
        bestn << std::flush;
    }

    auto result = history->Top();
    const auto& words = std::get<0>(result);

    std::string translation = Join((*vocab_)(words), " ", reverse_);

    best1 << translation;
    if(alignment_ > 0.f) {
      const auto& hypo = std::get<1>(result);
      auto align = getAlignment(hypo, alignment_);
      best1 << getAlignmentString(align);
    }
    best1 << std::flush;
  }

private:
  Ptr<Vocab> vocab_;
  bool reverse_{false};
  size_t nbest_{0};
  float alignment_{0.f};

  std::vector<HardAlignment> getAlignment(const Ptr<Hypothesis>& hyp,
                                          float threshold);
  std::string getAlignmentString(const std::vector<HardAlignment>& align);
};
}
