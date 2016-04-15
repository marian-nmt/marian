#pragma once

#include <string>
#include <fstream>
#include <iostream>
#include <set>
#include <map>
#include <iterator>

#include <boost/algorithm/string.hpp>

class BPE {
  using BPEPair = std::pair<std::string, std::string>;
 public:
  BPE(std::ifstream&& file, const std::string sep = "@@")
    : sep_(sep) {
    std::string inputLine;
    while (std::getline(file, inputLine)) {
      std::vector<std::string> code;
      boost::split(code, inputLine, boost::is_any_of(" "));
      bpeCodes_.emplace_back(code[0], code[1]);
    }
    //std::cerr << bpeCodes_.size() << " codes were loaded.\n";
  }

  BPE(const std::string& path, const std::string sep = "@@")
    : BPE(std::ifstream(path), sep) {}

  std::vector<std::string> Segment(const std::string& sentence) {
    std::vector<std::string> words, tokens;
    boost::split(words, sentence, boost::is_any_of(" "));
    //for (auto& word : words) std::cerr << word << " ";
    //std::cerr << std::endl;

    for (auto& word : words) {
      //std::cerr << "Encoding " << word << "..." << std::endl;
      if (word.empty()) continue;
      auto codes = Encode(word);
      for (size_t i = 0; i < codes.size() - 1; ++i) {
        tokens.emplace_back(codes[i] + sep_);
      }
      tokens.push_back(codes.back());
    }
    return tokens;
  }

  static std::set<BPEPair> GetPairs(std::vector<std::string>& word) {
    std::set<BPEPair> pairSet;
    for (size_t i = 1; i < word.size(); ++i) {
      pairSet.insert(std::make_pair(word[i-1], word[i]));
    }
    return pairSet;
  }

  BPEPair* FindBestBigram(const std::set<BPEPair>& pairs) {
    size_t minDist = bpeCodes_.size();

    for (const auto& pair : pairs) {
      auto it = std::find(bpeCodes_.begin(), bpeCodes_.end(), pair);
      size_t dist = std::distance(bpeCodes_.begin(), it);
      //std::cerr << "DIST: " << dist << " min: " << minDist << std::endl;
      if (dist < minDist) {
        minDist = dist;
      }
    }
    if (minDist == bpeCodes_.size()) {
      return nullptr;
    }
    else {
    return &bpeCodes_[minDist];
    }
  }

  std::vector<std::string> Encode(const std::string& word) {
    if (isCached(word)) {
      return cache_[word];
    }

    std::vector<std::string> vWord;
    for (auto& ch : word) {
      vWord.push_back(std::string(1, ch));
    }
    vWord.push_back("</w>");
    //std::cerr << "WORDS: ";
    // for (auto word : vWord) std::cerr << word << " ";
    //std::cerr << std::endl;

    auto pairs = GetPairs(vWord);
    //std::cerr << "PAIRS: ";
    // for (auto& pair : pairs) std::cerr << "(" << pair.first << " , " << pair.second <<")  " ;
    //std::cerr << std::endl;

    while (true) {
      BPEPair* bigram = FindBestBigram(pairs);
      if(bigram == nullptr) {
        break;
      }
      //std::cerr << "BEST BIGRAM: " << "(" << bigram->first << " < " << bigram->second << ")" << std::endl;

      std::vector<std::string> newWord;

      auto it = vWord.begin();
      while (it != vWord.end()) {
        auto jt = std::find(it, vWord.end(), bigram->first);
        for (auto i = it; i != jt; ++i) {
          newWord.push_back(*i);
        }

        if (jt == vWord.end()) {
          break;
        } else {
          it = jt;
        }

        if (*it == bigram->first && (it+1) != vWord.end() && *(it+1) == bigram->second) {
          newWord.emplace_back(bigram->first + bigram->second);
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
    if (vWord.back() == "</w>") {
      vWord.pop_back();
    }
    auto eos = vWord.back().find_last_of("</w>");
    if (eos != std::string::npos) {
      vWord.back().resize(eos - 3);
    }

    cache_[word] = vWord;
    //std::cerr << "RESULT: ";
    // for (auto& word : vWord) std:: cerr << word << " ";
    //std::cerr << std::endl;

    return vWord;
  }


 private:
  bool isCached(const std::string& word) {
    return cache_.find(word) != cache_.end();
  }

  std::vector<BPEPair> bpeCodes_;
  const std::string sep_;
  std::map<std::string, std::vector<std::string>> cache_;



};
