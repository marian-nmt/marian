#include "output_printer.h"

namespace marian {

data::WordAlignment OutputPrinter::getAlignment(const Ptr<Hypothesis>& hyp,
                                                float threshold) {
  data::SoftAlignment aligns;
  // Skip EOS
  auto last = hyp->GetPrevHyp();
  // Get soft alignments for each target word
  while(last->GetPrevHyp().get() != nullptr) {
    aligns.push_back(last->GetAlignment());
    last = last->GetPrevHyp();
  }

  return data::ConvertSoftAlignToHardAlign(aligns, threshold, true);
}

}  // namespace marian
