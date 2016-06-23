#include <cstdlib>
#include <iostream>
#include <string>
#include <boost/timer/timer.hpp>

#include "god.h"
#include "logging.h"
#include "search.h"
#include "printer.h"
#include "sentence.h"

#include <boost/python.hpp>

extern "C" 
{
  void openblas_set_num_threads(int num_threads);
}

void init(const std::string& options) {
  God::Init(options);
  size_t threadOpenBLAS = God::Get<size_t>("threads-openblas");
  LOG(info) << "Setting number of OpenBLAS threads to " << threadOpenBLAS;
  openblas_set_num_threads(threadOpenBLAS);
}

boost::python::list translate(boost::python::list& in) {
  thread_local std::unique_ptr<Search> search;
  if(!search) {
    LOG(info) << "Created Search for thread " << std::this_thread::get_id();
    search.reset(new Search(0));
  }
  boost::python::list result;
  for(int i = 0; i < boost::python::len(in); ++i) {
    std::stringstream ss;
    std::string s = boost::python::extract<std::string>(boost::python::object(in[i]));
    Printer(search->Decode(Sentence(i, s)), i, ss);
    result.append(ss.str());
  }
  return result;
}

BOOST_PYTHON_MODULE(libamunmt)
{
  boost::python::def("init", init);
  boost::python::def("translate", translate);
}
