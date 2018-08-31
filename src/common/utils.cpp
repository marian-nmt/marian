#include <stdio.h>
#include <array>
#include <boost/algorithm/string.hpp>
#include <iostream>
#include <sstream>

#include "3rd_party/exception.h"
#include "common/logging.h"
#include "common/utils.h"

namespace marian {
namespace utils {

void Trim(std::string& s) {
  boost::trim_if(s, boost::is_any_of(" \t\n"));
}

void Split(const std::string& line,
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

std::vector<std::string> Split(const std::string& line,
                               const std::string del /*= " "*/,
                               bool keepEmpty) {
  std::vector<std::string> pieces;
  Split(line, pieces, del, keepEmpty);
  return pieces;
}

// @TODO: SplitAny() shares all but 2 expressions with Split(). Merge them.
void SplitAny(const std::string& line,
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

std::vector<std::string> SplitAny(const std::string& line,
                                  const std::string del /*= " "*/,
                                  bool keepEmpty) {
  std::vector<std::string> pieces;
  SplitAny(line, pieces, del, keepEmpty);
  return pieces;
}

std::string Join(const std::vector<std::string>& words,
                 const std::string& del /*= " "*/,
                 bool reverse /*= false*/) {
  std::stringstream ss;
  if(words.empty()) {
    return "";
  }

  if(reverse) {
    for(size_t i = words.size() - 1; i > 0; --i) {
      ss << words[i] << del;
    }
    ss << words[0];
  } else {
    ss << words[0];
    for(size_t i = 1; i < words.size(); ++i) {
      ss << del << words[i];
    }
  }
  return ss.str();
}

std::string Exec(const std::string& cmd) {
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
