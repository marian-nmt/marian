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
           const std::string& del = " ",
           bool keepEmpty = false,
           bool anyOf = false);
void splitAny(const std::string& line,
              std::vector<std::string>& pieces,
              const std::string& del = " ",
              bool keepEmpty = false);

std::vector<std::string> split(const std::string& line,
                               const std::string& del = " ",
                               bool keepEmpty = false,
                               bool anyOf = false);
std::vector<std::string> splitAny(const std::string& line,
                                  const std::string& del = " ",
                                  bool keepEmpty = false);

std::string join(const std::vector<std::string>& words, const std::string& del = " ");

std::string exec(const std::string& cmd);

std::pair<std::string, int> hostnameAndProcessId();

std::string withCommas(size_t n);
bool endsWith(const std::string& text, const std::string& suffix);

std::string toUpper(const std::string& s);
double parseDouble(std::string s);
double parseNumber(std::string s);

}  // namespace utils
}  // namespace marian
