#pragma once

#include <string>
#include <fstream>
#include <iostream>
#include <set>
#include <map>

#include <boost/algorithm/string.hpp>



class BPE {
  using BPECode = std::vector<std::string>;
 public:
  BPE(std::ifstream&& file, const std::string sep = "@@")
    : sep_(sep) {
    std::string inputLine;
    while (std::getline(file, inputLine)) {
      BPECode code;
      boost::split(code, inputLine, boost::is_any_of(" "));
      bpeCodes_.push_back(code);
    }
  }

  BPE(const std::string& path, const std::string sep = "@@")
    : BPE(std::ifstream(path), sep) {}

  std::vector<std::string> Segment(const std::string& sentence) {
    std::vector<std::string> words, tokens;
    boost::split(words, sentence, boost::is_any_of(" "));

    for (auto& word : words) {
      auto codes = Encode(word);
      for (size_t i = 0; i < codes.size() - 1; ++i) {
        tokens.emplace_back(codes[i] + sep_);
      }
      tokens.push_back(codes.back());
    }
    return tokens;
  }

  template<class T>
  static std::set<std::pair<T,T>> GetPairs(const std::vector<T>& word) {
    std::set<std::pair<char, char>> pairSet;
    for (size_t i = 1; i < word.size(); ++i) {
      pairSet.insert(std::make_pair(word[i-1], word[i]));
    }
    return pairSet;
  }

  std::vector<std::string> Encode(const std::string& word) {
    if (isCached(word)) {
      return cache_[word];
    }

    std::vector<std::string> vWord(word.begin(), word.end());
    vWord.push_back("</w>");
    auto pairs = GetPairs(vWord);


    while (true) {
      std::vector<std::string> newWord;
      auto it = vWord.begin();
      std::pair<std::string, std::string> bigram;
      while (it != vWord.end()) {
        auto jt = std::find(it, vWord.end(), bigram.first);
        vWord.insert(vWord.end(), it, jt);

        if (jt == vWord.end()) {
          break;
        } else {
          it = jt;
        }

        if (*it == bigram.first && (it+1) != vWord.end() && *(it+1) == bigram.second) {
          newWord.emplace_back(bigram.first + bigram.second);
          it += 2;
        } else {
          newWord.push_back(*it);
          it += 1;
        }
      }
      vWord = newWord;
      if (vWord.size() == 1) {
        break;
      } else {
        pairs = GetPairs(vWord);
      }
    }
    if (vWord.back() == "</w>") vWord.pop_back();
    auto eos = std::find(vWord.back().begin(), vWord.back().end(), "</w>");
    vWord.back().erase(eos, vWord.back().end());

    cache_[word] = vWord;
    return vWord;
  }


 private:
  bool isCached(const std::string& word) {
    return cache_.find(word) != cache_.end();
  }
  std::vector<BPECode> bpeCodes_;
  const std::string sep_;
  std::map<std::string, std::vector<std::string>> cache_;



};
