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
}
}
