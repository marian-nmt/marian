#include "output_printer.h"

namespace marian {

data::WordAlignment OutputPrinter::getAlignment(const Ptr<Hypothesis>& hyp,
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

}  // namespace marian
