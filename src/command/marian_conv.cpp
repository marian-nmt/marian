#include "marian.h"

#include "common/cli_wrapper.h"

#include <sstream>

#include "graph/expression_graph_packable.h"

int main(int argc, char** argv) {
  using namespace marian;

  createLoggers();

  auto options = New<Options>();
  {
    auto cli = New<cli::CLIWrapper>(
        options,
        "Convert a model in the .npz format and normal memory layout to a mmap-able binary model which could be in normal memory layout or packed memory layout",
        "Allowed options",
        "Examples:\n"
        "  ./marian-conv -f model.npz -t model.bin --gemm-type fp16packed");
    cli->add<std::string>("--from,-f", "Input model", "model.npz");
    cli->add<std::string>("--to,-t", "Output model", "model.bin");
    cli->add<std::string>("--gemm-type,-g", "GEMM Type to be used with this weights", "mklfp32");
    cli->parse(argc, argv);
  }
  auto modelFrom = options->get<std::string>("from");
  auto modelTo = options->get<std::string>("to");
  auto saveGemmType = options->get<std::string>("gemm-type");

  LOG(info, "Outputting {}", modelTo);

  YAML::Node config;
  std::stringstream configStr;
  marian::io::getYamlFromModel(config, "special:model.yml", modelFrom);
  configStr << config;

  auto graph = New<ExpressionGraphPackable>();
  graph->setDevice(CPU0);
  graph->getBackend()->setOptimized(false);

  graph->load(modelFrom);
  graph->forward();
  // added a flag if the weights needs to be packed or not
  graph->packAndSave(modelTo, configStr.str(), /* --gemm-type */ saveGemmType, Type::float32);

  // graph->saveBinary(vm["bin"].as<std::string>());

  LOG(info, "Finished");

  return 0;
}
