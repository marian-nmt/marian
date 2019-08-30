#include "marian.h"

#include "common/cli_wrapper.h"

#include <sstream>

int main(int argc, char** argv) {
  using namespace marian;

  createLoggers();

  auto options = New<Options>();
  {
    auto cli = New<cli::CLIWrapper>(
        options,
        "Convert a model in the .npz format to a mmap-able binary model",
        "Allowed options",
        "Examples:\n"
        "  ./marian-conv -f model.npz -t model.bin");
    cli->add<std::string>("--from,-f", "Input model", "model.npz");
    cli->add<std::string>("--to,-t", "Output model", "model.bin");
    cli->parse(argc, argv);
  }
  auto modelFrom = options->get<std::string>("from");
  auto modelTo = options->get<std::string>("to");

  LOG(info, "Outputting {}", modelTo);

  YAML::Node config;
  std::stringstream configStr;
  marian::io::getYamlFromModel(config, "special:model.yml", modelFrom);
  configStr << config;

  auto graph = New<ExpressionGraph>(true);
  graph->setDevice(CPU0);
  graph->getBackend()->setOptimized(false);

  graph->load(modelFrom);
  graph->forward();
  graph->save(modelTo, configStr.str());

  // graph->saveBinary(vm["bin"].as<std::string>());

  LOG(info, "Finished");

  return 0;
}
