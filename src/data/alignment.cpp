#include "data/alignment.h"
#include "common/utils.h"

namespace marian {
namespace data {

WordAlignment::WordAlignment() {}

WordAlignment::WordAlignment(
    const std::vector<std::pair<size_t, size_t>>& align)
    : data_(align) {}

WordAlignment::WordAlignment(const std::string& line) {
  std::vector<std::string> atok = utils::Split(line, " -");
  for(size_t i = 0; i < atok.size(); i += 2)
    data_.emplace_back(std::stoi(atok[i]), std::stoi(atok[i + 1]));
}

void WordAlignment::sort() {
  std::sort(data_.begin(), data_.end(), [](const Point& a, const Point& b) {
    return (a.first == b.first) ? a.second < b.second : a.first < b.first;
  });
}

std::string WordAlignment::toString() const {
  std::stringstream str;
  for(auto p = begin(); p != end(); ++p) {
    if(p != begin())
      str << " ";
    str << p->first << "-" << p->second;
  }
  return str.str();
}

WordAlignment ConvertSoftAlignToHardAlign(SoftAlignment alignSoft,
                                          float threshold /*= 1.f*/,
                                          bool reversed /*= true*/,
                                          bool skipEOS /*= false*/) {
  size_t shift = alignSoft.size() > 0 && skipEOS ? 1 : 0;
  WordAlignment align;
  // Alignments by maximum value
  if(threshold == 1.f) {
    for(size_t t = 0; t < alignSoft.size() - shift; ++t) {
      // Retrieved alignments are in reversed order
      size_t rev = reversed ? alignSoft.size() - t - 1 : t;
      size_t maxArg = 0;
      for(size_t s = 0; s < alignSoft[0].size(); ++s) {
        if(alignSoft[rev][maxArg] < alignSoft[rev][s]) {
          maxArg = s;
        }
      }
      align.push_back(maxArg, t);
    }
  } else {
    // Alignments by greather-than-threshold
    for(size_t t = 0; t < alignSoft.size() - shift; ++t) {
      // Retrieved alignments are in reversed order
      size_t rev = reversed ? alignSoft.size() - t - 1 : t;
      for(size_t s = 0; s < alignSoft[0].size(); ++s) {
        if(alignSoft[rev][s] > threshold) {
          align.push_back(s, t);
        }
      }
    }
  }

  // Sort alignment pairs in ascending order
  align.sort();

  return align;
}

std::string SoftAlignToString(SoftAlignment align,
                              bool reversed /*= true*/,
                              bool skipEOS /*= false*/) {
  std::stringstream str;
  size_t shift = align.size() > 0 && skipEOS ? 1 : 0;
  bool first = true;
  for(size_t t = 0; t < align.size() - shift; ++t) {
    size_t rev = reversed ? align.size() - t - 1 : t;
    if(!first)
      str << " ";
    for(size_t s = 0; s < align[rev].size(); ++s) {
      if(s != 0)
        str << ",";
      str << align[rev][s];
    }
    first = false;
  }
  return str.str();
}

}  // namespace data
}  // namespace marian
