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

using namespace amunmt;

God god_;

std::shared_ptr<Histories> TranslationTask(const std::string& in, size_t taskCounter) {
  Search &search = god_.GetSearch();

  std::shared_ptr<Sentences> sentences(new Sentences());
  sentences->push_back(SentencePtr(new Sentence(god_, taskCounter, in)));
  return search.Process(god_, *sentences);
}

void init(const std::string& options) {
  god_.Init(options);
}

boost::python::list translate(boost::python::list& in) {
  size_t cpuThreads = god_.Get<size_t>("cpu-threads");
  LOG(info) << "Setting CPU thread count to " << cpuThreads;

  size_t totalThreads = cpuThreads;
#ifdef CUDA
  size_t gpuThreads = god_.Get<size_t>("gpu-threads");
  auto devices = god_.Get<std::vector<size_t>>("devices");
  LOG(info) << "Setting GPU thread count to " << gpuThreads;
  totalThreads += gpuThreads * devices.size();
#endif

  LOG(info) << "Total number of threads: " << totalThreads;
  amunmt_UTIL_THROW_IF2(totalThreads == 0, "Total number of threads is 0");

  ThreadPool pool(totalThreads);
  std::vector<std::future< std::shared_ptr<Histories> >> results;

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
    Printer(god_, *result.get().get(), ss);
    output.append(ss.str());
  }

  return output;
}

BOOST_PYTHON_MODULE(libamunmt)
{
  boost::python::def("init", init);
  boost::python::def("translate", translate);
}
