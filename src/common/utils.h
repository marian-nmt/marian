#pragma once

#include <string>
#include <vector>

namespace marian {
namespace utils {

void trim(std::string& s);
void trimLeft(std::string& s);
void trimRight(std::string& s);

void split(const std::string& line,
           std::vector<std::string>& pieces,
           const std::string del = " ",
           bool keepEmpty = false);

std::vector<std::string> split(const std::string& line,
                               const std::string del = " ",
                               bool keepEmpty = false);

void splitAny(const std::string& line,
              std::vector<std::string>& pieces,
              const std::string del = " ",
              bool keepEmpty = false);

std::vector<std::string> splitAny(const std::string& line,
                                  const std::string del = " ",
                                  bool keepEmpty = false);

std::string join(const std::vector<std::string>& words,
                 const std::string& del = " ",
                 bool reverse = false);

std::string exec(const std::string& cmd);

// wrapper around std::getline() that handles Windows input files with extra CR
// chars at the line end
template <class CharT, class Traits, class Allocator>
std::basic_istream<CharT, Traits>& getline(
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
std::basic_istream<CharT, Traits>& getline(
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
