#pragma once

#include <vector>

#include "common/god.h"
#include "common/history.h"
#include "common/utils.h"
#include "common/vocab.h"
#include "common/soft_alignment.h"

std::vector<size_t> GetAlignment(const HypothesisPtr& hypothesis);

template <class OStream>
void Printer(const History& history, OStream& out) {
  auto bestTranslation = history.Top();
  std::vector<std::string> bestSentenceWords = God::Postprocess(God::GetTargetVocab()(bestTranslation.first));

  std::string best;
  if (God::Get<bool>("return-alignment")) {
    auto alignment = GetAlignment(bestTranslation.second);
    best = Join(bestSentenceWords, alignment);
  } else {
    best = Join(bestSentenceWords);
  }
  LOG(progress) << "Best translation: " << best;

  if (God::Get<bool>("n-best")) {
    std::vector<std::string> scorerNames = God::GetScorerNames();
    const NBestList &nbl = history.NBest(God::Get<size_t>("beam-size"));
    if (God::Get<bool>("wipo")) {
      out << "OUT: " << nbl.size() << std::endl;
    }
    for (size_t i = 0; i < nbl.size(); ++i) {
      const Result& result = nbl[i];
      const Words &words = result.first;
      const HypothesisPtr &hypo = result.second;

      if(God::Get<bool>("wipo")) {
        out << "OUT: ";
      }
      std::string translation;
      if (God::Get<bool>("return-alignment")) {
        auto alignment = GetAlignment(bestTranslation.second);
        translation = Join(God::Postprocess(God::GetTargetVocab()(words)), alignment);
      } else {
        translation = Join(God::Postprocess(God::GetTargetVocab()(words)));
      }
      out << history.GetLineNum() << " ||| " << translation << " |||";
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
void Printer(const Histories& histories, OStream& out) {
  for (size_t i = 0; i < histories.size(); ++i) {
    const History& history = *histories.at(i).get();
    Printer(history, out);
  }
}
