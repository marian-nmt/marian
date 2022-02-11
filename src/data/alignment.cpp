#include "data/alignment.h"
#include "common/utils.h"

#include <algorithm>
#include <cmath>
#include <set>

namespace marian {
namespace data {

WordAlignment::WordAlignment() {}

WordAlignment::WordAlignment(const std::vector<Point>& align) : data_(align) {}

WordAlignment::WordAlignment(const std::string& line, size_t srcEosPos, size_t tgtEosPos) {
  std::vector<std::string> atok = utils::splitAny(line, " -");
  for(size_t i = 0; i < atok.size(); i += 2)
    data_.push_back(Point{ (size_t)std::stoi(atok[i]), (size_t)std::stoi(atok[i + 1]), 1.f });
  data_.push_back(Point{ srcEosPos, tgtEosPos, 1.f }); // add alignment point for both EOS symbols
}

void WordAlignment::sort() {
  std::sort(data_.begin(), data_.end(), [](const Point& a, const Point& b) {
    return (a.srcPos == b.srcPos) ? a.tgtPos < b.tgtPos : a.srcPos < b.srcPos;
  });
}

void WordAlignment::normalize(bool reverse/*=false*/) {
  std::vector<size_t> counts;
  counts.reserve(data_.size());
  
  // reverse==false : normalize target word prob by number of source words
  // reverse==true  : normalize source word prob by number of target words
  auto srcOrTgt = [](const Point& p, bool reverse) {
    return reverse ? p.srcPos : p.tgtPos;
  };

  for(const auto& a : data_) {
    size_t pos = srcOrTgt(a, reverse);
    if(counts.size() <= pos)
      counts.resize(pos + 1, 0);
    counts[pos]++;
  }
  
  // a.prob at this point is either 1 or normalized to a different value,
  // but we just set it to 1 / count, so multiple calls result in re-normalization
  // regardless of forward or reverse direction. We also set the remaining values to 1.
  for(auto& a : data_) {
    size_t pos = srcOrTgt(a, reverse);
    if(counts[pos] > 1)
      a.prob = 1.f / counts[pos];
    else 
      a.prob = 1.f;
  }
}

std::string WordAlignment::toString() const {
  std::stringstream str;
  for(auto p = begin(); p != end(); ++p) {
    if(p != begin())
      str << " ";
    str << p->srcPos << "-" << p->tgtPos;
  }
  return str.str();
}

WordAlignment ConvertSoftAlignToHardAlign(const SoftAlignment& alignSoft,
                                          float threshold /*= 1.f*/) {
  WordAlignment align;
  // Alignments by maximum value
  if(threshold == 1.f) {
    for(size_t t = 0; t < alignSoft.size(); ++t) {
      // Retrieved alignments are in reversed order
      size_t maxArg = 0;
      for(size_t s = 0; s < alignSoft[0].size(); ++s) {
        if(alignSoft[t][maxArg] < alignSoft[t][s]) {
          maxArg = s;
        }
      }
      align.push_back(maxArg, t, 1.f);
    }
  } else {
    // Alignments by greather-than-threshold
    for(size_t t = 0; t < alignSoft.size(); ++t) {
      // Retrieved alignments are in reversed order
      for(size_t s = 0; s < alignSoft[0].size(); ++s) {
        if(alignSoft[t][s] > threshold) {
          align.push_back(s, t, alignSoft[t][s]);
        }
      }
    }
  }
  // Sort alignment pairs in ascending order
  align.sort();

  return align;
}

std::string SoftAlignToString(SoftAlignment align) {
  std::stringstream str;
  bool first = true;
  for(size_t t = 0; t < align.size(); ++t) {
    if(!first)
      str << " ";
    for(size_t s = 0; s < align[t].size(); ++s) {
      if(s != 0)
        str << ",";
      str << align[t][s];
    }
    first = false;
  }
  return str.str();
}

}  // namespace data
}  // namespace marian
