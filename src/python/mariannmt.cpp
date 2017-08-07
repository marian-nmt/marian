#include <cstdlib>
#include <iostream>
#include <string>

#include <boost/python.hpp>

#include "common/version.h"
#include "translator/beam_search.h"
#include "translator/translator.h"

using namespace marian;


void init(const std::string& argopts) {
  std::cerr << "initialize...\n";
  auto options = New<Config>(argopts, ConfigMode::translating);
  std::cerr << "creating task...\n";
  auto task = New<TranslateMultiGPU<BeamSearch>>(options);
  std::cerr << "running...\n";
  task->run();
  std::cerr << "finished\n";
}

boost::python::list translate(boost::python::list& input) {
  boost::python::list output;
  for(int i = 0; i < boost::python::len(input); ++i)
    output.append(input[i]);
  return output;
}

std::string version() {
  return PROJECT_VERSION;
}


BOOST_PYTHON_MODULE(libmariannmt) {
  boost::python::def("init", init);
  boost::python::def("translate", translate);
  boost::python::def("version", version);
}
