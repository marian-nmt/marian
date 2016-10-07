#include "loader_factory.h"

#include "scorer.h"
#include "cpu/decoder/encoder_decoder_loader.h"

#ifdef CUDA
#include "gpu/decoder/encoder_decoder.h"
#include "gpu/decoder/ape_penalty.h"

#ifdef KENLM
#include "gpu/decoder/language_model.h"
#endif
#endif


LoaderPtr LoaderFactory::Create(
    const std::string& name,
    const YAML::Node& config,
    const std::string& mode) {
	Loader *loader;

  if (HAS_GPU_SUPPORT && (mode == "GPU")) {
    loader = CreateGPU(name, config);
    if (loader) {
      return LoaderPtr(loader);
    } else {
      LOG(info) << "No GPU scorer type. Switching to CPU";
    }
  }


	loader = CreateCPU(name, config);
	if (loader) {
		return LoaderPtr(loader);
	}

	std::string type = config["type"].as<std::string>();
	UTIL_THROW2("Unknown scorer in config file: " << type);
}

#ifdef CUDA
Loader *LoaderFactory::CreateGPU(
    const std::string& name,
    const YAML::Node& config) {
  UTIL_THROW_IF2(!config["type"],
				 "Missing scorer type in config file");

  std::string type = config["type"].as<std::string>();
  IF_MATCH_RETURN(type, "Nematus", GPU::EncoderDecoderLoader);
  IF_MATCH_RETURN(type, "nematus", GPU::EncoderDecoderLoader);
  IF_MATCH_RETURN(type, "NEMATUS", GPU::EncoderDecoderLoader);

  IF_MATCH_RETURN(type, "Ape", GPU::ApePenaltyLoader);
  IF_MATCH_RETURN(type, "ape", GPU::ApePenaltyLoader);
  IF_MATCH_RETURN(type, "APE", GPU::ApePenaltyLoader);

#ifdef KENLM
  IF_MATCH_RETURN(type, "KenLM", GPU::KenLMLoader)
  IF_MATCH_RETURN(type, "kenlm", GPU::KenLMLoader)
  IF_MATCH_RETURN(type, "KENLM", GPU::KenLMLoader)
#endif

  return NULL;
}
#endif


Loader *LoaderFactory::CreateCPU(const std::string& name,
						const YAML::Node& config) {
  UTIL_THROW_IF2(!config["type"],
				 "Missing scorer type in config file");
  std::string type = config["type"].as<std::string>();

  IF_MATCH_RETURN(type, "Nematus", CPU::EncoderDecoderLoader);
  IF_MATCH_RETURN(type, "nematus", CPU::EncoderDecoderLoader);
  IF_MATCH_RETURN(type, "NEMATUS", CPU::EncoderDecoderLoader);

  return NULL;
}
