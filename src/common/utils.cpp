#include "common/utils.h"
#include "common/logging.h"
#include "CLI/StringTools.hpp"

#include <stdio.h>
#include <array>
#include <iostream>
#include <sstream>
#include <string>
#ifdef __unix__
#include <unistd.h>
#endif

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

void split(const std::string& line,
           std::vector<std::string>& pieces,
           const std::string& del /*= " "*/,
           bool keepEmpty /*= false*/,
           bool anyOf /*= false*/) {
  size_t begin = 0;
  size_t pos = 0;
  std::string token;
  size_t delSize = anyOf ? 1 : del.size();
  while(true) {
    pos = anyOf ? line.find_first_of(del, begin) : line.find(del, begin);
    if(pos == std::string::npos)
      break;
    if(pos >= begin) {
      token = line.substr(begin, pos - begin);
      if(token.size() > 0 || keepEmpty)
        pieces.push_back(token);
    }
    begin = pos + delSize;
  }
  if(pos >= begin) {
    token = line.substr(begin, pos - begin);
    if(token.size() > 0 || keepEmpty)
      pieces.push_back(token);
  }
}

std::vector<std::string> split(const std::string& line,
                               const std::string& del /*= " "*/,
                               bool keepEmpty /*= false*/,
                               bool anyOf /*= false*/) {
  std::vector<std::string> pieces;
  split(line, pieces, del, keepEmpty, anyOf);
  return pieces;
}

void splitAny(const std::string& line,
              std::vector<std::string>& pieces,
              const std::string& del /*= " "*/,
              bool keepEmpty /*= false*/) {
  split(line, pieces, del, keepEmpty, /*anyOf =*/true);
}

std::vector<std::string> splitAny(const std::string& line,
                                  const std::string& del /*= " "*/,
                                  bool keepEmpty /*= false*/) {
  return split(line, del, keepEmpty, /*anyOf =*/true);
}

std::string join(const std::vector<std::string>& words, const std::string& del /*= " "*/) {
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

std::pair<std::string, int> hostnameAndProcessId() {  // helper to get hostname:pid
#ifdef _WIN32
  std::string hostname = getenv("COMPUTERNAME");
  auto processId = (int)GetCurrentProcessId();
#else
  static std::string hostname = []() {  // not sure if gethostname() is expensive.
                                        // This way we call it only once.
    char hostnamebuf[HOST_NAME_MAX + 1] = {0};
    gethostname(hostnamebuf, sizeof(hostnamebuf));
    return std::string(hostnamebuf);
  }();
  auto processId = (int)getpid();
#endif
  return {hostname, processId};
}

// format a long number with comma separators
std::string withCommas(size_t n) {
  std::string res = std::to_string(n);
  for(int i = (int)res.size() - 3; i > 0; i -= 3)
    res.insert(i, ",");
  return res;
}

bool endsWith(const std::string& text, const std::string& suffix) {
  return text.size() >= suffix.size()
         && !text.compare(text.size() - suffix.size(), suffix.size(), suffix);
}

std::string toUpper(const std::string& s) {
  std::locale loc;
  std::string res;
  res.reserve(s.capacity());
  for(auto c : s)  // @BUGBUG: This won't work with UTF-8 characters.
    res.push_back((char)std::toupper(c, loc));
  return res;
}

double parseDouble(std::string s) {
  double res;
  char c;  // dummy char -- if we succeed to parse this, then there were extraneous characters after the number
  auto rc = sscanf(s.c_str(), "%lf%c", &res, &c);
  ABORT_IF(rc != 1, "Mal-formed number: {}", s);
  return res;
}

// parses a user-friendly number that can have commas and (some) units
double parseNumber(std::string param) {
  // get unit prefix
  double factor = 1.;
  if(!param.empty() && param.back() >= 'A') {
    switch(param.back()) {
      case 'k': factor = 1.e3;  break;
      case 'M': factor = 1.e6;  break;
      case 'G': factor = 1.e9;  break;
      case 'T': factor = 1.e12; break;
      default: ABORT("Invalid or unsupported unit prefix '{}' in {}", param.back(), param);
    }
    param.pop_back();
  }
  // we allow users to place commas in numbers (note: we are not actually verifying that they are in
  // the right place)
  std::remove_if(param.begin(), param.end(), [](char c) { return c == ','; });
  return factor * parseDouble(param);
}

}  // namespace utils
}  // namespace marian
