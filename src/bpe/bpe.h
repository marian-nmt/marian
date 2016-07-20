#pragma once

#include <string>
#include <fstream>
#include <iostream>
#include <set>
#include <vector>
#include <unordered_map>
#include <map>
#include <iterator>

#include <boost/algorithm/string.hpp>

#include "utf8.h"

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

class BPE {
  using BPEPair = std::pair<std::string, std::string>;

 public:
  BPE(std::ifstream&& file, const std::string sep = "@@")
    : sep_(sep) {
    std::string inputLine;
    size_t index = 0;
    while (std::getline(file, inputLine)) {
      std::vector<std::string> code;
      boost::split(code, inputLine, boost::is_any_of(" "));
      bpeCodes_[make_pair(code[0], code[1])] = index++;
    }
  }

  BPE(const std::string& path, const std::string sep = "@@")
    : BPE(std::ifstream(path), sep) {}

  std::vector<std::string> Segment(const std::string& sentence) {
    std::vector<std::string> words, tokens;
    boost::split(words, sentence, boost::is_any_of(" "));

    for (auto& word : words) {
      if (word.empty()) continue;
      auto codes = Encode(word);
      for (size_t i = 0; i < codes.size() - 1; ++i) {
        tokens.emplace_back(codes[i] + sep_);
      }
      tokens.push_back(codes.back());
    }
    return tokens;
  }

  void PrintSegment(const std::string& sentence) {
    std::vector<std::string> words, tokens;
    boost::split(words, sentence, boost::is_any_of(" "));

    for (size_t wi = 0; wi < words.size(); ++wi) {
      if (words[wi].empty()) continue;
      auto codes = Encode(words[wi]);
      for (size_t i = 0; i < codes.size() - 1; ++i) {
        std::cout << codes[i] + sep_ << " ";
      }
      std::cout << codes.back();
      if (wi == words.size() -1 ) {
        std::cout << std::endl;
      } else {
        std::cout << " ";
      }
    }
  }

  static std::set<BPEPair> GetPairs(const std::vector<std::string>& word) {
    std::set<BPEPair> pairSet;
    for (size_t i = 1; i < word.size(); ++i) {
      pairSet.emplace(word[i-1], word[i]);
    }
    return pairSet;
  }

  const BPEPair* FindBestBigram(const std::set<BPEPair>& pairs) {
    size_t minDist = bpeCodes_.size();
    auto best = bpeCodes_.begin();

    for (const auto& pair : pairs) {
      auto it = bpeCodes_.find(pair);
      if (it == bpeCodes_.end()) {
        continue;
      }
      if (it->second < minDist) {
        minDist = it->second;
        best = it;
      }
    }
    if (minDist == bpeCodes_.size()) {
      return nullptr;
    }
    else {
    return &(best->first);
    }
  }

  std::vector<std::string>& Encode(const std::string& word) {
    if (isCached(word)) {
      return cache_[word];
    }

    std::vector<std::string> vWord = SplitWordIntoLetters(word);
    vWord.push_back("</w>");

    auto pairs = GetPairs(vWord);

    while (true) {
      const BPEPair* bigram = FindBestBigram(pairs);
      if (bigram == nullptr) {
        break;
      }

      std::vector<std::string> newWord;

      auto it = vWord.begin();
      while (it != vWord.end()) {
        auto jt = std::find(it, vWord.end(), bigram->first);
        for (auto i = it; i != jt; ++i) {
          newWord.push_back(*i);
        }

        if (jt == vWord.end()) {
          break;
        }
        it = jt;

        if (*it == bigram->first && (it+1) != vWord.end() && *(it+1) == bigram->second) {
          newWord.emplace_back(bigram->first + bigram->second);
          it += 2;
        } else {
          newWord.push_back(*it);
          it += 1;
        }
      }
      std::swap(vWord, newWord);
      if (newWord.size() == 1) {
        break;
      } else {
        pairs = GetPairs(vWord);
      }
    }

    if (vWord.back() == "</w>") {
      vWord.pop_back();
    }

    if (EndsWith(vWord.back(), "</w>")) {
      vWord.back().resize(vWord.back().size() - 4);
    }

    cache_[word] = vWord;

    return cache_[word];
  }


 private:
  bool isCached(const std::string& word) {
    return cache_.find(word) != cache_.end();
  }

  std::vector<std::string> SplitWordIntoLetters(const std::string& word) {
    char* charWord = (char*)word.c_str();
    auto b = charWord;
    auto e = charWord + strlen(charWord);

    std::vector<std::string> letters;
    int prevBegin = 0;
    while (b != e) {
      int end = utf8::next(b, e);
      std::vector<unsigned char> utf8result;
      utf8::utf32to8(&end,&end + 1, std::back_inserter(utf8result));
      letters.emplace_back(utf8result.begin(), utf8result.end());
    }
    return letters;
  }

  bool EndsWith(std::string const &fullString, std::string const suffix) {
    if (fullString.length() >= suffix.length()) {
      return (0 == fullString.compare(fullString.length() - suffix.length(), suffix.length(), suffix));
    } else {
      return false;
    }
  }

  std::unordered_map<BPEPair, size_t> bpeCodes_;
  const std::string sep_;
  std::unordered_map<std::string, std::vector<std::string>> cache_;
};


//class BPE {
//  public:
//    BPE(const std::string& sep = "@@ ")
//     : sep_(sep) {}
//    
//    std::string split(const std::string& line) {
//      return line;
//    }
//    
//    std::string unsplit(const std::string& line) {
//      std::string joined = line;
//      size_t pos = joined.find(sep_);
//      while(pos != std::string::npos) {
//        joined.erase(pos, sep_.size());
//        pos = joined.find(sep_, pos);
//      }
//      return joined;
//    }
//    
//    operator bool() const {
//      return false;
//    }
//    
//  private:
//    std::string sep_;
//};