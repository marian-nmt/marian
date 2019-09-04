#pragma once

#include <sstream>
#include <vector>

namespace marian {
namespace data {

class WordAlignment {
  struct Point
  {
      size_t srcPos;
      size_t tgtPos;
      float prob;
  };
private:
  std::vector<Point> data_;
public:
  WordAlignment();

  /**
   * @brief Constructs word alignments from a vector of pairs of two integers.
   *
   * @param align Vector of pairs of two unsigned integers
   */
private:
  WordAlignment(const std::vector<Point>& align);
public:

  /**
   * @brief Constructs word alignments from textual representation.
   *
   * @param line String in the form of "0-0 1-1 1-2", etc.
   */
  WordAlignment(const std::string& line);

  auto begin() const -> decltype(data_.begin()) { return data_.begin(); }
  auto end()   const -> decltype(data_.end())   { return data_.end(); }

  void push_back(size_t s, size_t t, float p) { data_.emplace_back(Point{ s, t, p }); }

  size_t size() const { return data_.size(); }

  /**
   * @brief Sorts alignments in place by source indices in ascending order.
   */
  void sort();

  /**
   * @brief Returns textual representation.
   */
  std::string toString() const;
};

// soft alignment = P(src pos|trg pos) for each beam and batch index, stored in a flattened CPU-side array
// Also used on QuickSAND boundary where beam and batch size is 1. Then it is simply [t][s] -> P(s|t)
typedef std::vector<std::vector<float>> SoftAlignment; // [trg pos][beam depth * max src length * batch size]

WordAlignment ConvertSoftAlignToHardAlign(SoftAlignment alignSoft,
                                          float threshold = 1.f);

std::string SoftAlignToString(SoftAlignment align);

}  // namespace data
}  // namespace marian
