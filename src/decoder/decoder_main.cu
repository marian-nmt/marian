#include <cstdlib>
#include <iostream>
#include <string>
#include <algorithm>
#include <memory>
#include <atomic>
#include <boost/timer/timer.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/lexical_cast.hpp>

#include "dl4mt.h"
#include "vocab.h"
#include "search.h"
#include "threadpool.h"

class BPE {
  public:
    BPE(const std::string& sep = "@@ ")
     : sep_(sep) {}
    
    std::string split(const std::string& line) {
      return line;
    }
    
    std::string unsplit(const std::string& line) {
      std::string joined = line;
      size_t pos = joined.find(sep_);
      while(pos != std::string::npos) {
        joined.erase(pos, sep_.size());
        pos = joined.find(sep_, pos);
      }
      return joined;
    }
    
    operator bool() const {
      return false;
    }
    
  private:
    std::string sep_;
};

int main(int argc, char* argv[]) {
  God::Init(argc, argv);
  
  std::cerr << "Translating...\n";
  std::ios_base::sync_with_stdio(false);

  BPE bpe;

  boost::timer::cpu_timer timer;
  
  ThreadPool pool(God::Get<size_t>("threads"));
  std::vector<std::future<History>> results;
  
  std::string in;
  std::size_t threadCounter = 0;
  while(std::getline(std::cin, in)) {
    auto call = [=, &bpe] {
      thread_local std::unique_ptr<Search> search;
      if(!search) {
        search.reset(new Search(threadCounter));
      }
      
      Vocab& srcVocab = God::GetSourceVocab();
      Sentence sentence = bpe ? srcVocab(bpe.split(in)) : srcVocab(in);
      return search->Decode(sentence);
    };
  
    results.emplace_back(
      pool.enqueue(call)
    );
    threadCounter++;
  }
  
  Vocab& trgVocab = God::GetTargetVocab();
  
  size_t lineCounter = 0;
  for(auto&& result : results) {
    History history = result.get();
      
    if(God::Get<bool>("n-best-list")) {
      NBestList nbl = history.NBest(God::Get<size_t>("beam-size"));
      for(size_t i = 0; i < nbl.size(); ++i) {
        auto& r = nbl[i];
        std::cout << lineCounter << " ||| " << (bpe ? bpe.unsplit(trgVocab(r.first)) : trgVocab(r.first)) << " |||";
        for(size_t j = 0; j < r.second->GetCostBreakdown().size(); ++j) {
          std::cout << " F" << j << "= " << r.second->GetCostBreakdown()[j];
        }
        std::cout << " ||| " << r.second->GetCost() << std::endl;
      }
    }
    else {
      std::string out = trgVocab(history.Top().first);
      if(bpe)
        out = bpe.unsplit(out);
      std::cout << out << std::endl;
    }
    history.Clear();
    lineCounter++;
  }
  std::cerr << timer.format() << std::endl;
  return 0;
}
