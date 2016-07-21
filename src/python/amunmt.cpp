#include <cstdlib>
#include <iostream>
#include <string>
#include <future>
#include <boost/thread/tss.hpp>
#include <boost/python.hpp>

#include "common/logging.h"
#include "common/threadpool.h"

#include "decoder/god.h"
#include "decoder/history.h"
#include "decoder/search.h"
#include "decoder/printer.h"
#include "decoder/sentence.h"

extern "C"
{
  void openblas_set_num_threads(int num_threads);
}

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
  size_t threadOpenBLAS = God::Get<size_t>("threads-openblas");
  LOG(info) << "Setting number of OpenBLAS threads to " << threadOpenBLAS;
  openblas_set_num_threads(threadOpenBLAS);
}

boost::python::list translate(boost::python::list& in) {
  size_t threadCount = God::Get<size_t>("threads");
  LOG(info) << "Setting number of threads to " << threadCount;
  ThreadPool pool(threadCount);
  std::vector<std::future<History>> results;
  std::size_t taskCounter = 0;

  boost::python::list result;
  for(int i = 0; i < boost::python::len(in); ++i) {
    std::string s = boost::python::extract<std::string>(boost::python::object(in[i]));
    results.emplace_back(
      pool.enqueue(
        [=]{ return TranslationTask(s, taskCounter); }
      )
    );
    ++taskCounter;
  }

  for (auto&& history : results) {
      result.append(God::GetTargetVocab()(history.get().Top().first));
  }
  return result;
}

BOOST_PYTHON_MODULE(libamunmt)
{
  boost::python::def("init", init);
  boost::python::def("translate", translate);
}
