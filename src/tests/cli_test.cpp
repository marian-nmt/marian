#include <iostream>
#include <iostream>
#include <memory>

#include "common/cli_wrapper.h"
#include "common/cli_helper.h"

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
  w.add<int>("-i,--int", "help message")->implicit_val("555")->default_val("123");
  w.add<std::string>("-s,--str", "help message")->default_val("foo");
  w.add<std::vector<float>>("-v,--vec", "help message")->expected(-2);
  w.switchGroup("My group");
  w.add<std::vector<std::string>>("--defvec,-d", "help message")->default_val("foo");
  w.add<bool>("-b,--bool", "help message");
  w.add<std::string>("--a-very-long-option-name-for-testing-purposes", "A very long text a very long text a very long text a very long text a very long text a very long text");
  w.switchGroup();
  w.add<std::string>("-f,--file", "help message")->check(validators::file_exists);
  //w.add<color>("-e,--enum", "help message for enum");

  auto opts = w.parse(argc, argv);

  opts["int"].as<int>();
  opts["str"].as<std::string>();
  opts["vec"].as<std::vector<float>>();
  opts["defvec"].as<std::vector<std::string>>();
  opts["bool"].as<bool>();
  //w.get<std::string>("long");
  opts["file"].as<std::string>();
  //w.get<color>("enum");

  YAML::Emitter emit;
  OutputYaml(w.getConfig(), emit);
  std::cout << emit.c_str() << std::endl;
  return 0;
}
