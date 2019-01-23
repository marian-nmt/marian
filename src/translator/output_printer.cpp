#include "output_printer.h"

namespace marian {

std::string OutputPrinter::getAlignment(const Ptr<Hypothesis>& hyp) {
  data::SoftAlignment align;
  auto last = hyp;
  // get soft alignments for each target word starting from the last one
  while(last->GetPrevHyp().get() != nullptr) {
    align.push_back(last->GetAlignment());
    last = last->GetPrevHyp();
  }

  // reverse alignments
  std::reverse(align.begin(), align.end());

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
}

}  // namespace marian
