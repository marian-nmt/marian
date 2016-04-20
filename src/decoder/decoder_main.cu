#include <cstdlib>
#include <iostream>
#include <string>
#include <boost/timer/timer.hpp>

#include "god.h"
#include "logging.h"
#include "search.h"
#include "threadpool.h"
#include "printer.h"

History TranslationTask(const std::string& in, size_t taskCounter) {
  thread_local std::unique_ptr<Search> search;
  if(!search) {
    LOG(info) << "Created Search for thread " << std::this_thread::get_id();  
    search.reset(new Search(taskCounter));
  }
  
  LOG(progress) << "Line " << taskCounter << " | Input: " << in;
  
  return search->Decode(God::GetSourceVocab()(in));  
}

int main(int argc, char* argv[]) {
  God::Init(argc, argv);
  boost::timer::cpu_timer timer;
  
  LOG(info) << "Reading input";
  
  std::string in;
  std::size_t taskCounter = 0;
  
  ThreadPool pool(God::Get<size_t>("threads"));
  std::vector<std::future<History>> results;
  while(std::getline(std::cin, in)) {
    
    results.emplace_back(
      pool.enqueue(
        [=]{ return TranslationTask(in, taskCounter); }
      )
    );
    
    taskCounter++;
  }
  
  size_t lineCounter = 0;
  for(auto&& result : results)
    Printer(result.get(), lineCounter++, std::cout);

  LOG(info) << timer.format();
  God::CleanUp();
  
  return 0;
}
