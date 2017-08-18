#include <cstdlib>
#include <iostream>
#include <string>

#include <boost/python.hpp>

#include "common/version.h"
#include "translator/beam_search.h"
#include "translator/translator.h"

using namespace marian;

Ptr<TranslateLoopMultiGPU<BeamSearch>> task;

void init(const std::string& argopts) {
  std::cerr << "initialize...\n";
  auto options = New<Config>(argopts, ConfigMode::translating);
  std::cerr << "creating task...\n";
  task = New<TranslateLoopMultiGPU<BeamSearch>>(options);
  std::cerr << "initialized...\n";
}

boost::python::list translate(boost::python::list& pyinput) {
  std::string inputText;
  for(int i = 0; i < boost::python::len(pyinput); ++i) {
    inputText += boost::python::extract<std::string>(
        boost::python::object(pyinput[i]));
    inputText += "\n";
  }

  std::vector<std::string> input = {inputText};
  std::vector<std::string> output = task->run(input);
  boost::python::list pyoutput;
  for(auto outputText : output) {
    pyoutput.append(outputText);
  }
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
