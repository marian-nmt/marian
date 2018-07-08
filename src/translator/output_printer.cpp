#include "output_printer.h"

namespace marian {

std::vector<HardAlignment> OutputPrinter::getAlignment(
    const Ptr<Hypothesis>& hyp,
    float threshold) {
  std::vector<SoftAlignment> alignSoft;
  // Skip EOS
  auto last = hyp->GetPrevHyp();
  // Get soft alignments for each target word
  while(last->GetPrevHyp().get() != nullptr) {
    alignSoft.push_back(last->GetAlignment());
    last = last->GetPrevHyp();
  }

  std::vector<HardAlignment> align;
  // Alignments by maximum value
  if(threshold == 1.f) {
    for(size_t t = 0; t < alignSoft.size(); ++t) {
      // Retrieved alignments are in reversed order
      size_t rev = alignSoft.size() - t - 1;
      size_t maxArg = 0;
      for(size_t s = 0; s < alignSoft[0].size(); ++s) {
        if(alignSoft[rev][maxArg] < alignSoft[rev][s]) {
          maxArg = s;
        }
      }
      align.push_back(std::make_pair(maxArg, t));
    }
  } else {
    // Alignments by greather-than-threshold
    for(size_t t = 0; t < alignSoft.size(); ++t) {
      // Retrieved alignments are in reversed order
      size_t rev = alignSoft.size() - t - 1;
      for(size_t s = 0; s < alignSoft[0].size(); ++s) {
        if(alignSoft[rev][s] > threshold) {
          align.push_back(std::make_pair(s, t));
        }
      }
    }
  }

  // Sort alignment pairs in ascending order
  std::sort(align.begin(),
            align.end(),
            [](const HardAlignment& a, const HardAlignment& b) {
              return (a.first == b.first) ? a.second < b.second
                                          : a.first < b.first;
            });

  return align;
}

std::string OutputPrinter::getAlignmentString(
    const std::vector<HardAlignment>& align) {
  std::stringstream alignStr;
  alignStr << " |||";
  for(auto p = align.begin(); p != align.end(); ++p) {
    alignStr << " " << p->first << "-" << p->second;
  }
  return alignStr.str();
}
}
