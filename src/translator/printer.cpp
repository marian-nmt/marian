#include "printer.h"

namespace marian {

std::vector<size_t> GetAlignment(const HypothesisPtr& hypothesis) {
  std::vector<SoftAlignment> aligns;
  HypothesisPtr last = hypothesis->GetPrevHyp();
  while(last->GetPrevHyp().get() != nullptr) {
    aligns.push_back(*(last->GetAlignment(0)));
    last = last->GetPrevHyp();
  }

  std::vector<size_t> alignment;
  for(auto it = aligns.rbegin(); it != aligns.rend(); ++it) {
    size_t maxArg = 0;
    for(size_t i = 0; i < it->size(); ++i) {
      if((*it)[maxArg] < (*it)[i]) {
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
  for(size_t wordIdx = 0; wordIdx < alignment.size(); ++wordIdx) {
    alignString << " " << wordIdx << "-" << alignment[wordIdx];
  }
  return alignString.str();
}
}
