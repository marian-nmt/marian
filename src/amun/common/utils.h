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
std::string Debug(const std::vector<T> &vec, size_t verbosity = 1)
{
  std::stringstream strm;

  strm << "size=" << vec.size();

  if (verbosity) {
    T sum = 0;
    for (size_t i = 0; i < vec.size(); ++i) {
      sum += vec[i];
    }
    strm << " sum=" << sum;
  }

  if (verbosity == 2) {
    for (size_t i = 0; i < vec.size(); ++i) {
      strm << " " << vec[i];
    }
  }

  return strm.str();
}

} // namespace

