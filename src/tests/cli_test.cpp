#include <iostream>
#include <iostream>
#include <memory>

#include "common/cli_wrapper.h"

using namespace marian::cli;

int main(int argc, char** argv) {
  enum color { red, green, blue };

  CLIWrapper w;
  w.add<int>("integer", "-i,--int", "help message for int", 555, true);
  w.add<std::string>("string", "-s,--str", "help message for str");
  w.add<std::vector<float>>("vector", "-v,--vec", "help message for vec")->expected(-3);
  w.add<bool>("bool", "-b,--bool", "help message for bool");
  //w.add<color>("enum", "-e,--enum", "help message for enum");

  //w.add("implicit", "-m,--implicit", "help message for implicit/bool");
  //w.add<int>("implicit", "-m,--implicit", "help message for implicit/val");
  w.parse(argc, argv);

  w.get<int>("integer");
  w.get<std::string>("string");
  w.get<std::vector<float>>("vector");
  w.get<bool>("bool");
  //w.get<color>("enum");

  return 0;
}
