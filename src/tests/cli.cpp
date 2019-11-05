#include <iostream>
#include <iostream>
#include <memory>

#include "common/definitions.h"
#include "common/options.h"
#include "common/cli_wrapper.h"
#include "common/cli_helper.h"

using namespace marian;
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

  auto options = New<Options>();
  {
    YAML::Node config;
    auto w = New<CLIWrapper>(config);
    w->add<int>("-i,--int", "help message")->implicit_val("555")->default_val("123");
    w->add<std::string>("-s,--str", "help message");
    w->add<std::vector<float>>("-v,--vec", "help message")->expected(-2);
    w->switchGroup("My group");
    w->add<std::vector<std::string>>("--defvec,-d", "help message")->default_val("foo");
    w->add<bool>("-b,--bool", "boolean option");
    w->add<bool>("-x,--xbool", "false boolean option", true);
    w->add<std::string>("--a-very-long-option-name-for-testing-purposes", "A very long text a very long text a very long text a very long text a very long text a very long text");
    w->switchGroup();
    //w->add<std::string>("-f,--file", "help message")->check(validators::file_exists);
    //w.add<color>("-e,--enum", "help message for enum");

    w->parse(argc, argv);
    options->merge(config);
  }

  options->get<int>("int");
  options->get<std::string>("str");
  options->get<std::vector<float>>("vec");
  options->get<std::vector<std::string>>("defvec");
  options->get<bool>("bool");
  //w.get<std::string>("long");
  options->get<std::string>("file");
  //w.get<color>("enum");

  YAML::Emitter emit;
  OutputYaml(options->cloneToYamlNode(), emit);
  std::cout << emit.c_str() << std::endl;

  std::cout << "===" << std::endl;
  std::cout << "vec/str.hasAndNotEmpty? " << options->hasAndNotEmpty("vec") << " " << options->hasAndNotEmpty("str") << std::endl;
  std::cout << "vec/str.has?      " << options->has("vec") << " " << options->has("str") << std::endl;

  return 0;
}
