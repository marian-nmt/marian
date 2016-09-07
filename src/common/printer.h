#pragma once

#include "god.h"
#include "common/history.h"

template <class OStream>
void Printer(const History& history, size_t lineNo, OStream& out) {
  std::string best = God::GetTargetVocab()(history.Top().first);
  LOG(progress) << "Best translation: " << best;
    
  if(God::Get<bool>("n-best")) {
    std::vector<std::string> scorerNames = God::GetScorerNames();
    const NBestList &nbl = history.NBest(God::Get<size_t>("beam-size"));
    for(size_t i = 0; i < nbl.size(); ++i) {
      const Result& result = nbl[i];
      const Words &words = result.first;
      const HypothesisPtr &hypo = result.second;

      out << lineNo << " ||| " << God::GetTargetVocab()(words) << " |||";
      for(size_t j = 0; j < hypo->GetCostBreakdown().size(); ++j) {
        out << " " << scorerNames[j] << "= " << hypo->GetCostBreakdown()[j];
      }
      if(God::Get<bool>("normalize")) {
        out << " ||| " << hypo->GetCost() / words.size() << std::endl;
      }
      else {
        out << " ||| " << hypo->GetCost() << std::endl;
      }
    }
  }
  else {
    out << best << std::endl;
  }
}
