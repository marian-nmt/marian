#include "output_printer.h"

#include <sstream>

namespace marian {

std::string OutputPrinter::getAlignment(const Hypothesis::PtrType& hyp) {
  data::SoftAlignment align;
  auto last = hyp;
  // get soft alignments for each target word starting from the last one
  while(last->getPrevHyp().get() != nullptr) {
    align.push_back(last->getAlignment());
    last = last->getPrevHyp();
  }

  // reverse alignments
  std::reverse(align.begin(), align.end());

  if(alignment_ == "soft") {
    return data::SoftAlignToString(align);
  } else if(alignment_ == "hard") {
    return data::ConvertSoftAlignToHardAlign(align, 1.f).toString();
  } else if(alignmentThreshold_ > 0.f) {
    return data::ConvertSoftAlignToHardAlign(align, alignmentThreshold_).toString();
  } else {
    ABORT("Unrecognized word alignment type");
  }
}

std::string OutputPrinter::getWordScores(const Hypothesis::PtrType& hyp) {
  std::ostringstream scores;
  scores.precision(5);
  for(const auto& score : hyp->tracebackWordScores())
    scores << " " << std::fixed << score;
  return scores.str();
}

}  // namespace marian
