#pragma once

#include <boost/algorithm/string.hpp>

namespace marian {
namespace data {

class WordAlignment {
private:
  typedef std::pair<int, int> Point;
  std::vector<Point> data_;

public:
  WordAlignment() {}

  /**
   * @brief Constructs the word alignment from its textual representation.
   *
   * @param line String in the form of "0-0 1-1 1-2", etc.
   */
  WordAlignment(const std::string& line) {
    std::vector<std::string> atok = split(line, " -");
    for(size_t i = 0; i < atok.size(); i += 2)
      data_.emplace_back(std::stoi(atok[i]), std::stoi(atok[i + 1]));
  }

  auto begin() const -> decltype(data_.begin()) { return data_.begin(); }
  auto end() const -> decltype(data_.end()) { return data_.end(); }

private:
  std::vector<std::string> split(const std::string& input,
                                 const std::string& chars) {
    std::vector<std::string> output;
    boost::split(output, input, boost::is_any_of(chars));
    return output;
  }
};

typedef std::vector<float> SoftAlignment;
typedef std::pair<size_t, size_t> HardAlignment;

static std::vector<HardAlignment> ConvertSoftAlignToHardAlign(
    std::vector<SoftAlignment> alignSoft,
    float threshold = .0f,
    bool reversed = true) {

  std::vector<data::HardAlignment> align;
  // Alignments by maximum value
  if(threshold == 1.f) {
    for(size_t t = 0; t < alignSoft.size(); ++t) {
      // Retrieved alignments are in reversed order
      size_t rev = alignSoft.size() - t - 1;
      size_t maxArg = 0;
      for(size_t s = 0; s < alignSoft[0].size(); ++s) {
        if(alignSoft[rev][maxArg] < alignSoft[rev][s]) {
          maxArg = s;
        }
      }
      align.push_back(std::make_pair(maxArg, t));
    }
  } else {
    // Alignments by greather-than-threshold
    for(size_t t = 0; t < alignSoft.size(); ++t) {
      // Retrieved alignments are in reversed order
      size_t rev = alignSoft.size() - t - 1;
      for(size_t s = 0; s < alignSoft[0].size(); ++s) {
        if(alignSoft[rev][s] > threshold) {
          align.push_back(std::make_pair(s, t));
        }
      }
    }
  }

  // Sort alignment pairs in ascending order
  std::sort(align.begin(),
            align.end(),
            [](const data::HardAlignment& a, const data::HardAlignment& b) {
              return (a.first == b.first) ? a.second < b.second
                                          : a.first < b.first;
            });

  return align;
}

}  // namespace data
}  // namespace marian
