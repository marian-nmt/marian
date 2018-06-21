#pragma once

#include <vector>
#include <iomanip>

#include "common/god.h"
#include "common/history.h"
#include "common/histories.h"
#include "common/utils.h"
#include "common/vocab.h"
#include "common/soft_alignment.h"
#include "common/sentence.h"
#include "common/sentences.h"

namespace amunmt {

std::vector<unsigned> GetAlignment(const HypothesisPtr& hypothesis);

std::string GetAlignmentString(const std::vector<unsigned>& alignment);
std::string GetSoftAlignmentString(const HypothesisPtr& hypothesis);
std::string GetNematusAlignmentString(const HypothesisPtr& hypothesis, std::string best, std::string source, unsigned linenum);

template <class OStream>
void Printer(const God &god, const History& history, OStream& out, const Sentence& sentence)
{
  if (sentence.size() == 0) {
    // empty line
    return;
  }

  auto bestTranslation = history.Top();
  std::vector<std::string> bestSentenceWords = god.Postprocess(god.GetTargetVocab()(bestTranslation.first));

  std::string best = Join(bestSentenceWords);
  if (god.Get<bool>("return-nematus-alignment")) {
	//Get the source sentence for printing Nematus style soft alignments
	std::string source = Join(god.Postprocess(god.GetSourceVocab()(sentence.GetWords(0))));
    best = GetNematusAlignmentString(bestTranslation.second, best, source, history.GetLineNum());
  }else{
    if (god.Get<bool>("return-alignment")) {
      best += GetAlignmentString(GetAlignment(bestTranslation.second));
    }
    if (god.Get<bool>("return-soft-alignment")) {
      best += GetSoftAlignmentString(bestTranslation.second);
    }
  }

  if (god.Get<bool>("n-best")) {
    std::vector<std::string> scorerNames = god.GetScorerNames();
    const NBestList &nbl = history.NBest(god.Get<unsigned>("beam-size"));
    if (god.Get<bool>("wipo")) {
      out << "OUT: " << nbl.size() << std::endl;
    }
    for (unsigned i = 0; i < nbl.size(); ++i) {
      const Result& result = nbl[i];
      const Words &words = result.first;
      const HypothesisPtr &hypo = result.second;

      if(god.Get<bool>("wipo")) {
        out << "OUT: ";
      }
      std::string translation = Join(god.Postprocess(god.GetTargetVocab()(words)));
      if (god.Get<bool>("return-alignment")) {
        translation += GetAlignmentString(GetAlignment(hypo));
      }
      if (god.Get<bool>("return-soft-alignment")) {
        translation += GetSoftAlignmentString(hypo);
      }
      out << history.GetLineNum() << " ||| " << translation << " |||";

      //std::cerr << "hypo->GetCostBreakdown().size()=" << hypo->GetCostBreakdown().size() << std::endl;
      for(unsigned j = 0; j < hypo->GetCostBreakdown().size(); ++j) {
        out << " " << scorerNames[j] << "= " << std::setprecision(3) << std::fixed << hypo->GetCostBreakdown()[j];
      }

      if(god.Get<bool>("normalize")) {
        out << " ||| " << std::setprecision(3) << std::fixed << hypo->GetCost() / words.size();
      }
      else {
        out << " ||| " << std::setprecision(3) << std::fixed << hypo->GetCost();
      }

      if(i < nbl.size() - 1)
        out << std::endl;
      else
        out << std::flush;

    }
  } else {
    out << best << std::flush;
  }
}

template <class OStream>
void Printer(const God &god, const Histories& histories, OStream& out, const Sentence& sentence) {
  for (unsigned i = 0; i < histories.size(); ++i) {
    const History& history = *histories.at(i).get();
    Printer(god, history, out, sentence);
  }
}

}

