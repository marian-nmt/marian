#include "output_printer.h"

namespace marian {

std::vector<data::HardAlignment> OutputPrinter::getAlignment(
    const Ptr<Hypothesis>& hyp,
    float threshold) {
  std::vector<data::SoftAlignment> alignSoft;
  // Skip EOS
  auto last = hyp->GetPrevHyp();
  // Get soft alignments for each target word
  while(last->GetPrevHyp().get() != nullptr) {
    alignSoft.push_back(last->GetAlignment());
    last = last->GetPrevHyp();
  }

  return data::ConvertSoftAlignToHardAlign(alignSoft, threshold, true);
}

std::string OutputPrinter::getAlignmentString(
    const std::vector<data::HardAlignment>& align) {
  std::stringstream alignStr;
  alignStr << " |||";
  for(auto p = align.begin(); p != align.end(); ++p) {
    alignStr << " " << p->first << "-" << p->second;
  }
  return alignStr.str();
}
}  // namespace marian
