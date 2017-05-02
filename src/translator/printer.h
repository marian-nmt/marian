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
             OStream& out) {
  if(options->has("n-best") && options->get<bool>("n-best")) {
    const auto &nbl = history->NBest(options->get<size_t>("beam-size"));
    
    for (size_t i = 0; i < nbl.size(); ++i) {
      const auto& result = nbl[i];
      const auto& words = result.first;
      const auto& hypo = result.second;

      std::string translation = Join((*vocab)(words));
      
      out << history->GetLineNum() << " ||| " << translation << " |||";
      
      if(hypo->GetCostBreakdown().empty()) {
        out << " F0=" << hypo->GetCost();  
      }
      else {
        for(size_t j = 0; j < hypo->GetCostBreakdown().size(); ++j) {
          out << " F" << j << "= " << hypo->GetCostBreakdown()[j];
        }
      }
      
      
      if(options->get<bool>("normalize")) {
        out << " ||| " << hypo->GetCost() / words.size();
      }
      else {
        out << " ||| " << hypo->GetCost();
      }
      
      if(i < nbl.size() - 1)
        out << std::endl;
      else
        out << std::flush;
      
    }
  }
  else {
    auto bestTranslation = history->Top();
    std::string translation = Join((*vocab)(bestTranslation.first));
    out << translation << std::flush;
  }
}

}

