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
  size_t threadCount = God::Get<size_t>("threads");
  LOG(info) << "Setting number of threads to " << threadCount;

  ThreadPool pool(threadCount);
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
