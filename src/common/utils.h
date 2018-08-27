#pragma once

#include <string>
#include <vector>

namespace marian {
namespace utils {

void Trim(std::string& s);

void Split(const std::string& line,
           std::vector<std::string>& pieces,
           const std::string del = " ",
           bool keepEmpty = false);

std::vector<std::string> Split(const std::string& line,
                               const std::string del = " ",
                               bool keepEmpty = false);

void SplitAny(const std::string& line,
              std::vector<std::string>& pieces,
              const std::string del = " ",
              bool keepEmpty = false);

std::vector<std::string> SplitAny(const std::string& line,
                                  const std::string del = " ",
                                  bool keepEmpty = false);

std::string Join(const std::vector<std::string>& words,
                 const std::string& del = " ",
                 bool reverse = false);

std::string Exec(const std::string& cmd);

// wrapper around std::getline() that handles Windows input files with extra CR
// chars at the line end
template <class CharT, class Traits, class Allocator>
std::basic_istream<CharT, Traits>& GetLine(
    std::basic_istream<CharT, Traits>& in,
    std::basic_string<CharT, Traits, Allocator>& line) {
  std::getline(in, line);
  // strip terminal CR if present
  if(in && !line.empty() && line.back() == in.widen('\r'))
    line.pop_back();
  return in;
}

// wrapper around std::getline() that handles Windows input files with extra CR
// chars at the line end
template <class CharT, class Traits, class Allocator>
std::basic_istream<CharT, Traits>& GetLine(
    std::basic_istream<CharT, Traits>& in,
    std::basic_string<CharT, Traits, Allocator>& line,
    CharT delim) {
  std::getline(in, line, delim);
  // strip terminal CR if present
  if(in && !line.empty() && line.back() == in.widen('\r'))
    line.pop_back();
  return in;
}

}  // namespace utils
}  // namespace marian
