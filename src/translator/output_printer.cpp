#include "output_printer.h"

namespace marian {

data::SoftAlignment OutputPrinter::getAlignment(const Ptr<Hypothesis>& hyp) {
  data::SoftAlignment aligns;
  // Skip EOS
  auto last = hyp->GetPrevHyp();
  // Get soft alignments for each target word
  while(last->GetPrevHyp().get() != nullptr) {
    aligns.push_back(last->GetAlignment());
    last = last->GetPrevHyp();
  }
  return aligns;
}

}  // namespace marian
