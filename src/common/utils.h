#pragma once

#include <string>
#include <vector>
#include <iostream>

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

// Split tab-separated line into the specified number of fields
void splitTsv(const std::string& line, std::vector<std::string>& fields, size_t numFields);

std::vector<std::string> split(const std::string& line,
                               const std::string& del = " ",
                               bool keepEmpty = false,
                               bool anyOf = false);
std::vector<std::string> splitAny(const std::string& line,
                                  const std::string& del = " ",
                                  bool keepEmpty = false);

// Return a string which is the concatenation of the strings in the given vector
std::string join(const std::vector<std::string>& words, const std::string& del = " ");
// Return a string which is the concatenation of values from the given vector of integers
std::string join(const std::vector<size_t>& words, const std::string& del = " ");
// Return a string which is the concatenation of values from the given vector of floats
std::string join(const std::vector<float>& words, const std::string& del = " ", size_t prec = 5);

std::string exec(const std::string& cmd, const std::vector<std::string>& args = {}, const std::string& arg = "");

std::pair<std::string, int> hostnameAndProcessId();

std::string withCommas(size_t n);
bool beginsWith(const std::string& text, const std::string& prefix);
bool endsWith(const std::string& text, const std::string& suffix);

std::string utf8ToUpper(const std::string& s);
std::string utf8ToLower(const std::string& s);
std::string utf8Capitalized(const std::string& word); // capitalize the first character only
std::string toEnglishTitleCase(const std::string& s);

std::u32string utf8ToUnicodeString(const std::string& s);
std::string utf8FromUnicodeString(const std::u32string& s);
std::u16string utf8ToUtf16String(const std::string& s);
std::string utf8FromUtf16String(const std::u16string& s);
bool isContinuousScript(char32_t c);

std::string findReplace(const std::string& in, const std::string& what, const std::string& withWhat, bool all = false);

double parseDouble(std::string s);
double parseNumber(std::string s);

// prints vector values with a custom label.
template<class T>
void Debug(const T *arr, size_t size, const std::string &str) {
  std::cerr << str << ":" << size << ": ";
  for (size_t i = 0; i < size; ++i) {
    std::cerr << arr[i] << " ";
  }
  std::cerr << std::endl;
}

template<class T>
void Debug(const std::vector<T> &arr, const std::string &str) {
  Debug(arr.data(), arr.size(), str);
}

}  // namespace utils
}  // namespace marian
