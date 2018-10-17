#include "common/utils.h"
#include "common/logging.h"
#include "3rd_party/exception.h"
#include "CLI/StringTools.hpp"

#include <stdio.h>
#include <array>
#include <iostream>
#include <sstream>

namespace marian {
namespace utils {

void trim(std::string& s) {
  CLI::detail::trim(s, " \t\n");
}

void trimRight(std::string& s) {
  CLI::detail::rtrim(s, " \t\n");
}

void trimLeft(std::string& s) {
  CLI::detail::ltrim(s, " \t\n");
}

// @TODO: use more functions from CLI instead of own implementations
void split(const std::string& line,
           std::vector<std::string>& pieces,
           const std::string del /*= " "*/,
           bool keepEmpty) {
  size_t begin = 0;
  size_t pos = 0;
  std::string token;
  while((pos = line.find(del, begin)) != std::string::npos) {
    if(pos >= begin) {
      token = line.substr(begin, pos - begin);
      if(token.size() > 0 || keepEmpty)
        pieces.push_back(token);
    }
    begin = pos + del.size();
  }
  if(pos >= begin) {
    token = line.substr(begin, pos - begin);
    if(token.size() > 0 || keepEmpty)
      pieces.push_back(token);
  }
}

std::vector<std::string> split(const std::string& line,
                               const std::string del /*= " "*/,
                               bool keepEmpty) {
  std::vector<std::string> pieces;
  split(line, pieces, del, keepEmpty);
  return pieces;
}

// @TODO: splitAny() shares all but 2 expressions with split(). Merge them.
void splitAny(const std::string& line,
              std::vector<std::string>& pieces,
              const std::string del /*= " "*/,
              bool keepEmpty) {
  size_t begin = 0;
  size_t pos = 0;
  std::string token;
  while((pos = line.find_first_of(del, begin)) != std::string::npos) {
    if(pos >= begin) {
      token = line.substr(begin, pos - begin);
      if(token.size() > 0 || keepEmpty)
        pieces.push_back(token);
    }
    begin = pos + 1;
  }
  if(pos >= begin) {
    token = line.substr(begin, pos - begin);
    if(token.size() > 0 || keepEmpty)
      pieces.push_back(token);
  }
}

std::vector<std::string> splitAny(const std::string& line,
                                  const std::string del /*= " "*/,
                                  bool keepEmpty) {
  std::vector<std::string> pieces;
  splitAny(line, pieces, del, keepEmpty);
  return pieces;
}

std::string join(const std::vector<std::string>& words,
                 const std::string& del /*= " "*/) {
  std::stringstream ss;
  if(words.empty()) {
    return "";
  }

  ss << words[0];
  for(size_t i = 1; i < words.size(); ++i) {
    ss << del << words[i];
  }

  return ss.str();
}

std::string exec(const std::string& cmd) {
  std::array<char, 128> buffer;
  std::string result;
#ifdef _WIN32
#define popen _popen
#define pclose _pclose
#endif
  std::shared_ptr<std::FILE> pipe(popen(cmd.c_str(), "r"), pclose);
  if(!pipe)
    ABORT("popen() failed!");

  while(!std::feof(pipe.get())) {
    if(std::fgets(buffer.data(), 128, pipe.get()) != NULL)
      result += buffer.data();
  }
  return result;
}

}  // namespace utils
}  // namespace marian
