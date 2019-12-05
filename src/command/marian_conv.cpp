#include "marian.h"

#include "common/cli_wrapper.h"

#include <sstream>

#include "tensors/cpu/fbgemm/expression_graph_packable.h"

int main(int argc, char** argv) {
  using namespace marian;

  createLoggers();

  auto options = New<Options>();
  {
    YAML::Node config; // @TODO: get rid of YAML::Node here entirely to avoid the pattern. Currently not fixing as it requires more changes to the Options object.
    auto cli = New<cli::CLIWrapper>(
        config,
        "Convert a model in the .npz format and normal memory layout to a mmap-able binary model which could be in normal memory layout or packed memory layout",
        "Allowed options",
        "Examples:\n"
        "  ./marian-conv -f model.npz -t model.bin --gemm-type packed16");
    cli->add<std::string>("--from,-f", "Input model", "model.npz");
    cli->add<std::string>("--to,-t", "Output model", "model.bin");
    cli->add<std::string>("--gemm-type,-g", "GEMM Type to be used: float32, packed16, packed8", "float32");
    cli->parse(argc, argv);
    options->merge(config);
  }
  auto modelFrom = options->get<std::string>("from");
  auto modelTo = options->get<std::string>("to");
  
  auto saveGemmTypeStr = options->get<std::string>("gemm-type", "float32");
  Type saveGemmType;
  if(saveGemmTypeStr == "float32") {
    saveGemmType = Type::float32;
  } else if(saveGemmTypeStr == "packed16") {
    saveGemmType = Type::packed16;
  } else if(saveGemmTypeStr == "packed8") {
    saveGemmType = Type::packed8;
  } else {
    ABORT("Unknown gemm-type: {}", saveGemmTypeStr);
  }

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
