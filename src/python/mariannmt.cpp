#include <cstdlib>
#include <iostream>
#include <string>

#include <boost/python.hpp>

#include "common/utils.h"
#include "common/version.h"
#include "translator/beam_search.h"
#include "translator/translator.h"

using namespace marian;

Ptr<TranslateServiceMultiGPU<BeamSearch>> task;

void init(const std::string& argopts) {
  auto options = New<Config>(argopts, ConfigMode::translating);
  task = New<TranslateServiceMultiGPU<BeamSearch>>(options);
  LOG(info)->info("Translator initialized");
}

boost::python::list translate(boost::python::list& pyinput) {
  std::vector<std::string> input;
  for(int i = 0; i < boost::python::len(pyinput); ++i) {
    input.emplace_back(
        boost::python::extract<std::string>(boost::python::object(pyinput[i])));
  }

  auto output = task->run(input);

  boost::python::list pyoutput;
  pyoutput.append(Join(output, "\n"));
  return pyoutput;
}

std::string version() {
  return PROJECT_VERSION;
}

BOOST_PYTHON_MODULE(libmariannmt) {
  boost::python::def("init", init);
  boost::python::def("translate", translate);
  boost::python::def("version", version);
}
