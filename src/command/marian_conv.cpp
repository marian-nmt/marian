#include "marian.h"

#include "common/cli_wrapper.h"

#include <sstream>

int main(int argc, char** argv) {
  using namespace marian;

  createLoggers();

  cli::CLIWrapper cli("Allowed options");
  cli.add<std::string>("--from,-f", "Input model", "model.npz");
  cli.add<std::string>("--to,-t", "Output model", "model.bin");

  try {
    cli.app()->parse(argc, argv);
  } catch(const CLI::ParseError& e) {
    exit(cli.app()->exit(e));
  }

  auto modelFrom = cli.get<std::string>("from");
  auto modelTo = cli.get<std::string>("to");

  LOG(info, "Outputting {}", modelTo);

  YAML::Node config;
  std::stringstream configStr;
  marian::io::getYamlFromModel(config, "special:model.yml", modelFrom);
  configStr << config;

  auto graph = New<ExpressionGraph>(true, false);
  graph->setDevice(CPU0);

  graph->load(modelFrom);
  graph->forward();
  graph->save(modelFrom, configStr.str());

  // graph->saveBinary(vm["bin"].as<std::string>());

  LOG(info, "Finished");

  return 0;
}
