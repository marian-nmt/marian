#include <cstdlib>
#include <iostream>
#include <string>
#include <boost/timer/timer.hpp>

#include "god.h"
#include "search.h"
#include "threadpool.h"
#include "printer.h"

int main(int argc, char* argv[]) {
  God::Init(argc, argv);
  std::ios_base::sync_with_stdio(false);
  boost::timer::cpu_timer timer;
  
  std::cerr << "Translating...\n";
  
  std::string in;
  std::size_t taskCounter = 0;
  
  ThreadPool pool(God::Get<size_t>("threads"));
  std::vector<std::future<History>> results;
  while(std::getline(std::cin, in)) {
      
    auto translationTask = [in, taskCounter] {
      thread_local std::unique_ptr<Search> search;
      if(!search)
        search.reset(new Search(taskCounter));
      return search->Decode(God::GetSourceVocab()(in));
    };
    
    results.emplace_back(pool.enqueue(translationTask));
    
    taskCounter++;
  }
  
  size_t lineCounter = 0;
  for(auto&& result : results)
    Printer(result.get(), lineCounter++, std::cout);

  std::cerr << timer.format() << std::endl;
  God::CleanUp();
  
  return 0;
}
