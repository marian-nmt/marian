#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <set>
#include <unordered_map>
#include <iterator>

#include "common/processor/processor.h"


template<class T>
inline void hash_combine(std::size_t & seed, const T & v)
{
  std::hash<T> hasher;
  seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

namespace std
{
  template<typename S, typename T> struct hash<pair<S, T>>
  {
      inline size_t operator()(const pair<S, T> & v) const
      {
            size_t seed = 0;
            ::hash_combine(seed, v.first);
            ::hash_combine(seed, v.second);
            return seed;
          }
    };
}

namespace amunmt {

class BPE : public Processor {
  using BPEPair = std::pair<std::string, std::string>;

  public:
    BPE();
    BPE(std::ifstream&& file, const std::string sep = "@@");

    BPE(const std::string& path, const std::string sep = "@@");

    std::vector<std::string> Segment(const std::string& sentence);

    void PrintSegment(const std::string& sentence);

    std::vector<std::string>& Encode(const std::string& word) const;

    std::vector<std::string> Encode(const std::vector<std::string>& words) const;

    std::vector<std::string> Preprocess(const std::vector<std::string> input) const;
    std::vector<std::string> Postprocess(const std::vector<std::string> input) const;

    virtual ~BPE() {}
  private:
    std::set<BPEPair> GetPairs(const std::vector<std::string>& word) const;

    const BPEPair* FindBestBigram(const std::set<BPEPair>& pairs) const;

    bool IsCached(const std::string& word) const;

    std::vector<std::string> SplitWordIntoLetters(const std::string& word) const;

    bool EndsWith(const std::string& fullString, const std::string suffix) const;

    std::unordered_map<BPEPair, size_t> bpeCodes_;
    const std::string sep_;
    mutable std::unordered_map<std::string, std::vector<std::string>> cache_;


};
}
