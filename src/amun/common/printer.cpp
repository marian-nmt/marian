#include "printer.h"

namespace amunmt {

std::vector<size_t> GetAlignment(const HypothesisPtr& hypothesis) {
  std::vector<SoftAlignment> aligns;
  HypothesisPtr last = hypothesis->GetPrevHyp();
  while (last->GetPrevHyp().get() != nullptr) {
    aligns.push_back(*(last->GetAlignment(0)));
    last = last->GetPrevHyp();
  }

  std::vector<size_t> alignment;
  for (auto it = aligns.rbegin(); it != aligns.rend(); ++it) {
    size_t maxArg = 0;
    for (size_t i = 0; i < it->size(); ++i) {
      if ((*it)[maxArg] < (*it)[i]) {
        maxArg = i;
      }
    }
    alignment.push_back(maxArg);
  }

  return alignment;
}


std::string GetAlignmentString(const std::vector<size_t>& alignment) {
  std::stringstream alignString;
  alignString << " |||";
  for (size_t wordIdx = 0; wordIdx < alignment.size(); ++wordIdx) {
    alignString << " " << wordIdx << "-" << alignment[wordIdx];
  }
  return alignString.str();
}

std::string GetSoftAlignmentString(const HypothesisPtr& hypothesis) {
  std::vector<SoftAlignment> aligns;
  HypothesisPtr last = hypothesis->GetPrevHyp();
  while (last->GetPrevHyp().get() != nullptr) {
    aligns.push_back(*(last->GetAlignment(0)));
    last = last->GetPrevHyp();
  }

  std::stringstream alignString;
  alignString << " |||";
  for (auto it = aligns.rbegin(); it != aligns.rend(); ++it) {
    alignString << " ";
    for (size_t i = 0; i < it->size(); ++i) {
      if (i>0) alignString << ",";
      alignString << (*it)[i];
    }
    // alternate code: distribute probability mass from alignment to <eos>
    // float aligned_to_eos = (*it)[it->size()-1];
    // for (size_t i = 0; i < it->size()-1; ++i) {
    //  if (i>0) alignString << ",";
    //  alignString << ( (*it)[i] / (1-aligned_to_eos) );
    // }
  }

  return alignString.str();
}

std::string GetNematusAlignmentString(const HypothesisPtr& hypothesis, std::string best, std::string source, size_t linenum) {
  std::vector<SoftAlignment> aligns;
  HypothesisPtr last = hypothesis;
  while (last->GetPrevHyp().get() != nullptr) {
    aligns.push_back(*(last->GetAlignment(0)));
    last = last->GetPrevHyp();
  }
  //<Sentence Number> ||| <Translation> ||| 0 ||| <Source> ||| <Source word count> <Translation word count>
  std::stringstream firstline;
  int srcspaces = std::count_if(source.begin(), source.end(), [](unsigned char c){ return std::isspace(c); });
  
  firstline << linenum << " ||| " << best << " ||| " << hypothesis->GetCost() / aligns.size() * -1 
  << " ||| " << source << " ||| " << srcspaces+2 << " " << aligns.size();

  std::stringstream alignString;
  for (auto it = aligns.rbegin(); it != aligns.rend(); ++it) {
    alignString << "\n";
    for (size_t i = 0; i < srcspaces+2; ++i) {
      if (i>0) alignString << " ";
      alignString << (*it)[i];
    }
  }
  alignString << "\n";

  return firstline.str() + alignString.str();
}

}

