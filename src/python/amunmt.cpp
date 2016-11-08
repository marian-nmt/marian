#include <cstdlib>
#include <iostream>
#include <string>
#include <boost/timer/timer.hpp>
#include <boost/thread/tss.hpp>
#include <boost/python.hpp>

#include "common/god.h"
#include "common/logging.h"
#include "common/threadpool.h"
#include "common/search.h"
#include "common/printer.h"
#include "common/sentence.h"
#include "common/exception.h"

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

void init(const std::string& options) {
  God::Init(options);
}

boost::python::list translate(boost::python::list& in) {
  size_t cpuThreads = God::Get<size_t>("cpu-threads");
  LOG(info) << "Setting CPU thread count to " << cpuThreads;

  size_t totalThreads = cpuThreads;
#ifdef CUDA
  size_t gpuThreads = God::Get<size_t>("gpu-threads");
  auto devices = God::Get<std::vector<size_t>>("devices");
  LOG(info) << "Setting GPU thread count to " << gpuThreads;
  totalThreads += gpuThreads * devices.size();
#endif

  LOG(info) << "Total number of threads: " << totalThreads;
  UTIL_THROW_IF2(totalThreads == 0, "Total number of threads is 0");

  ThreadPool pool(totalThreads);
  std::vector<std::future<History>> results;

  boost::python::list output;
  for(int i = 0; i < boost::python::len(in); ++i) {
    std::string s = boost::python::extract<std::string>(boost::python::object(in[i]));
    results.emplace_back(
        pool.enqueue(
            [=]{ return TranslationTask(s, i); }
        )
    );
  }

  size_t lineCounter = 0;

  for (auto&& result : results) {
    std::stringstream ss;
    Printer(result.get(), lineCounter++, ss);
    output.append(ss.str());
  }

  return output;
}

BOOST_PYTHON_MODULE(libamunmt)
{
  boost::python::def("init", init);
  boost::python::def("translate", translate);
}
