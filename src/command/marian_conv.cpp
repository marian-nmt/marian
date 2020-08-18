#include "marian.h"

#include "common/cli_wrapper.h"

#include <sstream>

#include "tensors/cpu/fbgemm/expression_graph_packable.h"
#include "onnx/expression_graph_onnx_exporter.h"

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
    cli->add<std::string>("--export-as", "Kind of conversion: marian-bin or onnx-{encode,decoder-step,decoder-init,decoder-stop}", "marian-bin");
    cli->add<std::string>("--gemm-type,-g", "GEMM Type to be used: float32, packed16, packed8avx2, packed8avx512", "float32");
    cli->add<std::vector<std::string>>("--vocabs,-V", "Vocabulary file, required for ONNX export");
    cli->parse(argc, argv);
    options->merge(config);
  }
  auto modelFrom = options->get<std::string>("from");
  auto modelTo = options->get<std::string>("to");

  auto exportAs = options->get<std::string>("export-as");
  auto vocabPaths = options->get<std::vector<std::string>>("vocabs");// , std::vector<std::string>());
  
  auto saveGemmTypeStr = options->get<std::string>("gemm-type", "float32");
  Type saveGemmType;
  if(saveGemmTypeStr == "float32") {
    saveGemmType = Type::float32;
  } else if(saveGemmTypeStr == "packed16") {  // packed16 only supports AVX2. AVX512 might be added later
    saveGemmType = Type::packed16;
  } else if(saveGemmTypeStr == "packed8avx2") { // packed8 for AVX2
    saveGemmType = Type::packed8avx2;
  } else if(saveGemmTypeStr == "packed8avx512") { // packed8 for AVX512
    saveGemmType = Type::packed8avx512;
  } else {
    ABORT("Unknown gemm-type: {}", saveGemmTypeStr);
  }

  LOG(info, "Outputting {}, precision: {}", modelTo, saveGemmType);

  YAML::Node config;
  std::stringstream configStr;
  marian::io::getYamlFromModel(config, "special:model.yml", modelFrom);
  configStr << config;

  auto load = [&](Ptr<ExpressionGraph> graph) {
    graph->setDevice(CPU0);
    graph->getBackend()->setOptimized(false);

    graph->load(modelFrom);
    graph->forward();  // run the initializers
  };

  if (exportAs == "marian-bin") {
    auto graph = New<ExpressionGraphPackable>();
    load(graph);
    // added a flag if the weights needs to be packed or not
    graph->packAndSave(modelTo, configStr.str(), /* --gemm-type */ saveGemmType, Type::float32);
  }
  else if (exportAs == "onnx-encode") {
#ifdef USE_ONNX
    auto graph = New<ExpressionGraphONNXExporter>();
    load(graph);
    auto modelOptions = New<Options>(config)->with("vocabs", vocabPaths, "inference", true);

    graph->exportToONNX(modelTo, modelOptions, vocabPaths);
#else
    ABORT("--export-as onnx-encode requires Marian to be built with USE_ONNX=ON");
#endif // USE_ONNX
  }
  else
    ABORT("Unknown --export-as value: {}", exportAs);

  // graph->saveBinary(vm["bin"].as<std::string>());

  LOG(info, "Finished");

  return 0;
}
