#include <cstdlib>
#include <iostream>
#include <string>
#include <boost/timer/timer.hpp>

#ifdef __APPLE__
#include <boost/thread/tss.hpp>
#endif

#include "common/logging.h"
#include "common/threadpool.h"

#include "decoder/god.h"
#include "decoder/printer.h"
#include "decoder/search.h"
#include "decoder/sentence.h"

#ifdef PROFILER_MODE
#include <gperftools/profiler.h>
#endif

History TranslationTask(const std::string& in, size_t taskCounter) {
  #ifdef __APPLE__
    static boost::thread_specific_ptr<Search> s_search;
    Search *search = s_search.get();

    if(search == NULL) {
      LOG(info) << "Created Search for thread " << std::this_thread::get_id();
      search = new Search(taskCounter);
      s_search.reset(search);
    }
  #else
    thread_local std::unique_ptr<Search> search;

    if(!search) {
      LOG(info) << "Created Search for thread " << std::this_thread::get_id();
      search.reset(new Search(taskCounter));
    }
  #endif

  return search->Decode(Sentence(taskCounter, in));
}

int main(int argc, char* argv[]) {
  God::Init(argc, argv);
  boost::timer::cpu_timer timer;

  std::string in;
  std::size_t taskCounter = 0;

  size_t threadCount = God::Get<size_t>("threads");
  LOG(info) << "Setting number of threads to " << threadCount;
  ThreadPool pool(threadCount);
  std::vector<std::future<History>> results;

#ifdef PROFILER_MODE
  ProfilerStart("amun_profile.log");
#endif

  LOG(info) << "Reading input";
  while(std::getline(God::GetInputStream(), in)) {

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

#ifdef PROFILER_MODE
  ProfilerStop();
#endif

  LOG(info) << "Total time: " << timer.format();
  God::CleanUp();

  return 0;
}
