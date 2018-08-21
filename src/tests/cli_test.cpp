#include <iostream>
#include <iostream>
#include <memory>

#include "common/cli_wrapper.h"

using namespace marian::cli;

enum color { red, green, blue };

template <typename S>
S &operator<<(S &s, const color &v) {
  if(v == color::red)
    s << "red";
  else if(v == color::green)
    s << "green";
  else if(v == color::blue)
    s << "blue";
  return s;
}

template <typename S>
S &operator>>(S &s, color &r) {
  std::string v;
  s >> v;
  if(v == "red")
    r = color::red;
  else if(v == "green")
    r = color::blue;
  else if(v == "blue")
    r = color::blue;
}

int main(int argc, char** argv) {

  CLIWrapper w;
  w.add<int>("integer", "-i,--int", "help message for int")->implicit_val("555");
  w.add<std::string>("string", "-s,--str", "help message for str")->default_val("foo");
  w.add<std::vector<float>>("vector", "-v,--vec", "help message for vec")->expected(-2);
  w.add<bool>("bool", "-b,--bool", "help message for bool");
  w.add<color>("enum", "-e,--enum", "help message for enum");

  try {
    w.parse(argc, argv);
  } catch(const CLI::ParseError& e) {
    return w.app()->exit(e);
  }

  w.get<int>("integer");
  w.get<std::string>("string");
  w.get<std::vector<float>>("vector");
  w.get<bool>("bool");
  w.get<color>("enum");

  return 0;
}
