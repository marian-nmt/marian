#include "printer.h"

namespace marian {

std::vector<size_t> GetAlignment(const Ptr<Hypothesis>& hyp) {
  typedef std::vector<float> SoftAlignment;

  std::vector<SoftAlignment> aligns;
  auto last = hyp->GetPrevHyp();
  while(last->GetPrevHyp().get() != nullptr) {
    aligns.push_back(last->GetAlignment());
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

std::string GetAlignmentString(const std::vector<size_t>& align) {
  std::stringstream alignStr;
  alignStr << " |||";
  for(size_t wIdx = 0; wIdx < align.size(); ++wIdx) {
    alignStr << " " << wIdx << "-" << align[wIdx];
  }
  return alignStr.str();
}

}
