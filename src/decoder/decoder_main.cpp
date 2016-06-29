#include <cstdlib>
#include <iostream>
#include <string>
#include <boost/timer/timer.hpp>
#include <boost/thread/tss.hpp>

#include "god.h"
#include "logging.h"
#include "search.h"
#include "threadpool.h"
#include "printer.h"
#include "sentence.h"

History TranslationTask(const std::string& in, size_t taskCounter) {
  static boost::thread_specific_ptr<Search> s_search;
  Search *search = s_search.get();
  if(search == NULL) {
    LOG(info) << "Created Search for thread " << std::this_thread::get_id();
	search = new Search(taskCounter);
    s_search.reset(search);
  }
  
  return search->Decode(Sentence(taskCounter, in));  
}

extern "C" 
{
  void openblas_set_num_threads(int num_threads);
}

int main(int argc, char* argv[]) {
  God::Init(argc, argv);
  boost::timer::cpu_timer timer;
  
  std::string in;
  std::size_t taskCounter = 0;
  
  size_t threadOpenBLAS = God::Get<size_t>("threads-openblas");
  
  LOG(info) << "Setting number of OpenBLAS threads to " << threadOpenBLAS;
  openblas_set_num_threads(threadOpenBLAS);
  
  size_t threadCount = God::Get<size_t>("threads");
  LOG(info) << "Setting number of threads to " << threadCount;
  ThreadPool pool(threadCount);
  std::vector<std::future<History>> results;
  
  LOG(info) << "Reading input";
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

  LOG(info) << "Total time: " << timer.format();
  God::CleanUp();
  
  return 0;
}
