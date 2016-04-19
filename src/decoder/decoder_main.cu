#include <cstdlib>
#include <iostream>
#include <string>
#include <boost/timer/timer.hpp>

#include "dl4mt.h"
#include "vocab.h"
#include "search.h"
#include "threadpool.h"
#include "printer.h"

int main(int argc, char* argv[]) {
  God::Init(argc, argv);
  
  std::cerr << "Translating...\n";
  std::ios_base::sync_with_stdio(false);
  boost::timer::cpu_timer timer;
  
  ThreadPool pool(God::Get<size_t>("threads"));
  std::vector<std::future<History>> results;
  
  std::string in;
  std::size_t threadCounter = 0;
  while(std::getline(std::cin, in)) {
    auto call = [=] {
      thread_local std::unique_ptr<Search> search;
      if(!search)
        search.reset(new Search(threadCounter));
      return search->Decode(God::GetSourceVocab()(in));
    };
  
    results.emplace_back(
      pool.enqueue(call)
    );
    threadCounter++;
  }
  
  size_t lineCounter = 0;
  for(auto&& result : results) {
    History history = result.get();
    Printer(history, lineCounter, std::cout);
    lineCounter++;
  }
  std::cerr << timer.format() << std::endl;
  return 0;
}
