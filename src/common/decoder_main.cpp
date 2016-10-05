#include <cstdlib>
#include <iostream>
#include <string>
#include <boost/timer/timer.hpp>
#include <boost/thread/tss.hpp>

#include "common/god.h"
#include "common/logging.h"
#include "common/search.h"
#include "common/threadpool.h"
#include "common/printer.h"
#include "common/sentence.h"

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
  LOG(info) << "Initialization... DONE";
  std::setvbuf(stdout, NULL, _IONBF, 0);
  boost::timer::cpu_timer timer;

  std::string in;
  std::size_t taskCounter = 0;

  size_t threadCount;
  if (God::Get<std::string>("mode") == "GPU") {
    threadCount= God::Get<size_t>("threads-per-device")
                 * God::Get<std::vector<size_t>>("devices").size();
  } else {
    threadCount = God::Get<size_t>("threads");
  }

  LOG(info) << "threadCount set to " << threadCount;

  if (God::Get<bool>("wipo")) {
    LOG(info) << "Reading input";
    while (std::getline(God::GetInputStream(), in)) {
      History result = TranslationTask(in, taskCounter);
      Printer(result, taskCounter++, std::cout);
    }
  } else {
    LOG(info) << "Setting number of threads to " << threadCount;
    ThreadPool pool(threadCount);
    LOG(info) << "Reading input";

    std::vector<std::future<History>> results;

    while(std::getline(God::GetInputStream(), in)) {

      results.emplace_back(
        pool.enqueue(
          [=]{ return TranslationTask(in, taskCounter); }
        )
      );

      taskCounter++;
    }

    size_t lineCounter = 0;
    for (auto&& result : results)
      Printer(result.get(), lineCounter++, std::cout);
  }
  LOG(info) << "Total time: " << timer.format();
  God::CleanUp();

  return 0;
}
