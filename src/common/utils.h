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
                 const std::string& del = " ");

std::string exec(const std::string& cmd);

std::pair<std::string, int> hostnameAndProcessId();

}  // namespace utils
}  // namespace marian
