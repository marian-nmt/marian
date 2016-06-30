#pragma once

#include "god.h"

template <class OStream>
void Printer(const History& history, size_t lineNo, OStream& out) {
  //std::cerr << history << std::endl;
  std::string best = God::GetTargetVocab()(history.Top().first);
  LOG(progress) << "Best translation: " << best;

  if(God::Get<bool>("n-best")) {
    std::vector<std::string> scorerNames = God::GetScorerNames();
    NBestList nbl = history.NBest(God::Get<size_t>("beam-size"));
    for(size_t i = 0; i < nbl.size(); ++i) {
      auto& r = nbl[i];
      out << lineNo << " ||| " << God::GetTargetVocab()(r.first) << " |||";
      for(size_t j = 0; j < r.second->GetCostBreakdown().size(); ++j) {
        out << " " << scorerNames[j] << "= " << r.second->GetCostBreakdown()[j];
      }
      if(God::Get<bool>("normalize"))
        out << " ||| " << r.second->GetCost() / r.first.size() << std::endl;
      else
        out << " ||| " << r.second->GetCost() << std::endl;
    }
  }
  else {
    out << best << std::endl;
  }
}
