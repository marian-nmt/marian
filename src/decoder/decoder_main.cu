#include <cstdlib>
#include <iostream>
#include <string>
#include <boost/timer/timer.hpp>

#include "god.h"
#include "backend_gpu.h"
#include "logging.h"
#include "search.h"
#include "threadpool.h"
#include "printer.h"
#include "sentence.h"

template <class Backend>
History TranslationTask(const std::string& in, size_t taskCounter) {
  thread_local std::unique_ptr<Search<Backend>> search;
  if(!search) {
    LOG(info) << "Created Search for thread " << std::this_thread::get_id();
    search.reset(new Search<Backend>(taskCounter));
  }
  
  return search->Decode(Sentence(taskCounter, in));  
}

int main(int argc, char* argv[]) {
  God::Init(argc, argv);
  boost::timer::cpu_timer timer;
  
  std::string in;
  std::size_t taskCounter = 0;
  
  size_t threadCount = God::Get<size_t>("threads-per-device")
                       * God::Get<std::vector<size_t>>("devices").size();
  LOG(info) << "Setting number of threads to " << threadCount;
  ThreadPool pool(threadCount);
  std::vector<std::future<History>> results;
  
  LOG(info) << "Reading input";
  while(std::getline(std::cin, in)) {
    
    results.emplace_back(
      pool.enqueue(
        [=]{ return TranslationTask<BackendGPU>(in, taskCounter); }
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
