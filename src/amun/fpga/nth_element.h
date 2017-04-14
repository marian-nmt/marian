#pragma once

#include <vector>
#include <algorithm>


namespace amunmt {
namespace FPGA {

class NthElement {
public:
  NthElement() = delete;
  NthElement(const NthElement &copy) = delete;
  NthElement(size_t maxBeamSize, size_t maxBatchSize);

};

} // namespace
}
