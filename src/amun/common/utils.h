#pragma once
#include <sstream>
#include <string>
#include <vector>
#include <boost/algorithm/string.hpp>

namespace amunmt {

void Trim(std::string& s);

void Split(const std::string& line, std::vector<std::string>& pieces, const std::string del=" ");

std::string Join(const std::vector<std::string>& words, const std::string del=" ");
std::string Join(const std::vector<std::string>& words,
                 const std::vector<size_t>& align, const std::string del=" ");


////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
std::string Debug(const std::vector<T> &vec, unsigned verbosity = 1)
{
  std::stringstream strm;

  strm << "size=" << vec.size();

  if (verbosity) {
    T sum(0);
    for (unsigned i = 0; i < vec.size(); ++i) {
      sum += vec[i];
    }
    strm << " sum=" << sum;
  }

  if (verbosity == 2) {
    for (unsigned i = 0; i < vec.size(); ++i) {
      strm << " " << vec[i];
    }
  }

  return strm.str();
}

template<>
inline std::string Debug(const std::vector<char> &vec, unsigned verbosity)
{
  std::stringstream strm;

  strm << "size=" << vec.size();

  if (verbosity) {
    unsigned sum(0);
    for (unsigned i = 0; i < vec.size(); ++i) {
      sum += vec[i] ? 1 : 0;
    }
    strm << " sum=" << sum;
  }

  if (verbosity == 2) {
    for (unsigned i = 0; i < vec.size(); ++i) {
      strm << " " << (vec[i] ? 1 : 0);
    }
  }

  return strm.str();
}

} // namespace

