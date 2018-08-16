#include "output_printer.h"

namespace marian {

std::string OutputPrinter::getAlignment(const Ptr<Hypothesis>& hyp) {
  data::SoftAlignment align;
  // Skip EOS
  auto last = hyp->GetPrevHyp();
  // Get soft alignments for each target word
  while(last->GetPrevHyp().get() != nullptr) {
    align.push_back(last->GetAlignment());
    last = last->GetPrevHyp();
  }

  if(alignment_ == "soft") {
    return data::SoftAlignToString(align);
  } else if(alignment_ == "hard") {
    return data::ConvertSoftAlignToHardAlign(align, 1.f).toString();
  } else if(alignmentThreshold_ > 0.f) {
    return data::ConvertSoftAlignToHardAlign(align, alignmentThreshold_)
        .toString();
  } else {
    ABORT("Unrecognized word alignment type");
  }

  return "";
}

}  // namespace marian
