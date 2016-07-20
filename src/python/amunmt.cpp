#include <cstdlib>
#include <iostream>
#include <string>
#include <boost/thread/tss.hpp>
#include <boost/python.hpp>

#include "common/logging.h"

#include "decoder/god.h"
#include "decoder/search.h"
#include "decoder/printer.h"
#include "decoder/sentence.h"

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
  static boost::thread_specific_ptr<Search> s_search;
  Search *search = s_search.get();

  if(search == NULL) {
    LOG(info) << "Created Search for thread " << std::this_thread::get_id();
   	search = new Search(0);
    s_search.reset(search);
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
