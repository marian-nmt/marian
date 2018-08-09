#pragma once

#include <boost/algorithm/string.hpp>
#include <sstream>

namespace marian {
namespace data {

class WordAlignment {
private:
  typedef std::pair<int, int> Point;
  std::vector<Point> data_;

public:
  /**
   * @brief Constructs word alignments from a vector of pairs of two integers.
   *
   * @param align Vector of pairs of two unsigned integers
   */
  WordAlignment(const std::vector<std::pair<int, int>>& align) : data_(align) {}

  /**
   * @brief Constructs word alignments from textual representation.
   *
   * @param line String in the form of "0-0 1-1 1-2", etc.
   */
  WordAlignment(const std::string& line) {
    std::vector<std::string> atok = split(line, " -");
    for(size_t i = 0; i < atok.size(); i += 2)
      data_.emplace_back(std::stoi(atok[i]), std::stoi(atok[i + 1]));
  }

  WordAlignment() {}

  auto begin() const -> decltype(data_.begin()) { return data_.begin(); }
  auto end() const -> decltype(data_.end()) { return data_.end(); }

  void push_back(size_t s, size_t t) { data_.push_back(std::make_pair(s, t)); }

  /**
   * @brief Sorts alignments in place by source indices in ascending order.
   */
  void sort() {
    std::sort(data_.begin(), data_.end(), [](const Point& a, const Point& b) {
      return (a.first == b.first) ? a.second < b.second : a.first < b.first;
    });
  }

  std::string toString() const {
    std::stringstream str;
    for(auto p = begin(); p != end(); ++p) {
      if(p != begin())
        str << " ";
      str << p->first << "-" << p->second;
    }
    return str.str();
  }

private:
  std::vector<std::string> split(const std::string& input,
                                 const std::string& chars) const {
    std::vector<std::string> output;
    boost::split(output, input, boost::is_any_of(chars));
    return output;
  }
};

typedef std::vector<std::vector<float>> SoftAlignment;

static WordAlignment ConvertSoftAlignToHardAlign(SoftAlignment alignSoft,
                                                 float threshold = 1.f,
                                                 bool reversed = true) {
  WordAlignment align;
  // Alignments by maximum value
  if(threshold == 1.f) {
    for(size_t t = 0; t < alignSoft.size(); ++t) {
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
    for(size_t t = 0; t < alignSoft.size(); ++t) {
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

}  // namespace data
}  // namespace marian
