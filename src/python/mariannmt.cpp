#include <cstdlib>
#include <iostream>
#include <string>

#include <boost/python.hpp>

#include "common/version.h"


void init(const std::string& options) {
  // TODO: implement me!
}


boost::python::list translate(boost::python::list& in) {
  // TODO: implement me!

  boost::python::list output;
  output.append("foo");
  output.append("bar");
  output.append("baz");

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
