#include <cstdlib>
#include <iostream>
#include <string>
#include <boost/timer/timer.hpp>

#include "god.h"
#include "logging.h"
#include "search.h"
#include "printer.h"
#include "sentence.h"

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

std::string translate(const std::string& in) {
  thread_local std::unique_ptr<Search> search;
  if(!search) {
    LOG(info) << "Created Search for thread " << std::this_thread::get_id();
    search.reset(new Search(0));
  }
  std::stringstream ss;
  Printer(search->Decode(Sentence(0, in)), 0, ss);
  return ss.str();
}

#include <boost/python.hpp>

BOOST_PYTHON_MODULE(libamunmt)
{
  boost::python::def("init", init);
  boost::python::def("translate", translate);
}
