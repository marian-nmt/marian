#pragma once

#include <vector>

#include "common/god.h"
#include "common/history.h"
#include "common/utils.h"
#include "common/vocab.h"
#include "common/soft_alignment.h"

template <class OStream>
void Printer(const History& history, size_t lineNo, OStream& out) {
  std::string best = Join(God::Postprocess(God::GetTargetVocab()(history.Top().first)));
  LOG(progress) << "Best translation: " << best;

  // if (God::Get<bool>("return-alignment")) {
    // auto last = history.Top().second;
    // std::vector<SoftAlignment> aligns;
    // while (last->GetPrevHyp().get() != nullptr) {
      // aligns.push_back(*(last->GetAlignment(0)));
      // last = last->GetPrevHyp();
    // }
    // std::stringstream ss;
    // for (auto it = aligns.rbegin(); it != aligns.rend(); ++it) {
      // ss << "(";
      // for (auto sIt = it->begin(); sIt != it->end(); ++sIt) {
        // ss << *sIt << " ";
      // }
      // ss << ") | ";
    // }
    // LOG(progress) << "ALIGN: " << ss.str();
  // }

  if(God::Get<bool>("n-best")) {
    std::vector<std::string> scorerNames = God::GetScorerNames();
    const NBestList &nbl = history.NBest(God::Get<size_t>("beam-size"));
    if(God::Get<bool>("wipo")) {
      out << "OUT: " << nbl.size() << std::endl;
    }
    for(size_t i = 0; i < nbl.size(); ++i) {
      const Result& result = nbl[i];
      const Words &words = result.first;
      const HypothesisPtr &hypo = result.second;

      if(God::Get<bool>("wipo"))
        out << "OUT: ";
      out << lineNo << " ||| " << Join(God::Postprocess(God::GetTargetVocab()(words))) << " |||";
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

template <class OStream>
void Printer(const Histories& histories, size_t lineNo, OStream& out) {

  for (const History& history: histories) {
    Printer(history, lineNo, out);
  }
}
