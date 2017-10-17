#pragma once

#include <vector>

#include "common/utils.h"
#include "data/vocab.h"
#include "translator/history.h"

namespace marian {

template <class OStream>
void Printer(Ptr<Config> options,
             Ptr<Vocab> vocab,
             Ptr<History> history,
             OStream& best1,
             OStream& bestn) {
  if(options->has("n-best") && options->get<bool>("n-best")) {
    const auto& nbl = history->NBest(options->get<size_t>("beam-size"));

    for(size_t i = 0; i < nbl.size(); ++i) {
      const auto& result = nbl[i];
      const auto& words = std::get<0>(result);
      const auto& hypo = std::get<1>(result);

      float realCost = std::get<2>(result);

      std::string translation = Join((*vocab)(words));

      bestn << history->GetLineNum() << " ||| " << translation << " |||";

      if(hypo->GetCostBreakdown().empty()) {
        bestn << " F0=" << hypo->GetCost();
      } else {
        for(size_t j = 0; j < hypo->GetCostBreakdown().size(); ++j) {
          bestn << " F" << j << "= " << hypo->GetCostBreakdown()[j];
        }
      }

      bestn << " ||| " << realCost;

      if(i < nbl.size() - 1)
        bestn << std::endl;
      else
        bestn << std::flush;
    }
  }

  auto bestTranslation = history->Top();
  std::string translation = Join((*vocab)(std::get<0>(bestTranslation)));
  best1 << translation << std::flush;
}
}
