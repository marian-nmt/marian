#include "marian.h"
#include "common/cli_wrapper.h"
#include "tensors/cpu/expression_graph_packable.h"
#include "onnx/expression_graph_onnx_exporter.h"
#include "layers/lsh.h"

#include <sstream>

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
    cli->add<std::string>("--gemm-type,-g", "GEMM Type to be used: float32, packed16, packed8avx2, packed8avx512, "
                          "intgemm8, intgemm8ssse3, intgemm8avx2, intgemm8avx512, intgemm16, intgemm16sse2, intgemm16avx2, intgemm16avx512", 
                          "float32");
    cli->add<std::vector<std::string>>("--add-lsh", 
                                       "Encode output matrix and optional rotation matrix into model file. "
                                       "arg1: number of bits in LSH encoding, arg2: name of output weights matrix")->implicit_val("1024 Wemb");
    cli->add<std::vector<std::string>>("--vocabs,-V", "Vocabulary file, required for ONNX export");
    cli->parse(argc, argv);
    options->merge(config);
  }
  auto modelFrom = options->get<std::string>("from");
  auto modelTo = options->get<std::string>("to");

  auto exportAs = options->get<std::string>("export-as");
  auto vocabPaths = options->get<std::vector<std::string>>("vocabs");// , std::vector<std::string>());

  bool addLsh = options->hasAndNotEmpty("add-lsh");
  int lshNBits = 1024;
  std::string lshOutputWeights = "Wemb";
  if(addLsh) {
    auto lshParams   = options->get<std::vector<std::string>>("add-lsh");
    lshNBits         = std::stoi(lshParams[0]);
    if(lshParams.size() > 1)
      lshOutputWeights = lshParams[1];
  }
  
  // We accept any type here and will later croak during packAndSave if the type cannot be used for conversion
  Type saveGemmType = typeFromString(options->get<std::string>("gemm-type", "float32"));

  LOG(info, "Outputting {}, precision: {}", modelTo, saveGemmType);

  YAML::Node config;
  std::stringstream configStr;
  marian::io::getYamlFromModel(config, "special:model.yml", modelFrom);
  configStr << config;

  if (exportAs == "marian-bin") {
    auto graph = New<ExpressionGraphPackable>();
    graph->setDevice(CPU0);
    graph->load(modelFrom);

    if(addLsh) {
      // Add dummy parameters for the LSH before the model gets actually initialized.
      // This create the parameters with useless values in the tensors, but it gives us the memory we need.
      graph->setReloaded(false);
      lsh::addDummyParameters(graph, /*weights=*/lshOutputWeights, /*nBits=*/lshNBits);
      graph->setReloaded(true);
    }

    graph->forward();  // run the initializers

    if(addLsh) {
      // After initialization, hijack the paramters for the LSH and force-overwrite with correct values.
      // Once this is done we can just pack and save as normal.
      lsh::overwriteDummyParameters(graph, /*weights=*/lshOutputWeights);
    }

    // added a flag if the weights needs to be packed or not
    graph->packAndSave(modelTo, configStr.str(), /* --gemm-type */ saveGemmType, Type::float32);
  }
  else if (exportAs == "onnx-encode") {
#ifdef USE_ONNX
    auto graph = New<ExpressionGraphONNXExporter>();
    graph->setDevice(CPU0);
    graph->load(modelFrom);
    graph->forward();  // run the initializers
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
