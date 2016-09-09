#include "loader_factory.h"
#include "scorer.h"
#include "gpu/decoder/encoder_decoder.h"
#include "gpu/decoder/ape_penalty.h"

#ifdef KENLM
#include "gpu/decoder/language_model.h"
#endif

LoaderPtr LoaderFactory::Create(const std::string& name,
						const YAML::Node& config)
{
	Loader *loader;
	loader = CreateGPU(name, config);
	if (loader) {
		return LoaderPtr(loader);
	}

	loader = CreateCPU(name, config);
	if (loader) {
		return LoaderPtr(loader);
	}

	std::string type = config["type"].as<std::string>();
	UTIL_THROW2("Unknown scorer in config file: " << type);
}

Loader *LoaderFactory::CreateGPU(const std::string& name,
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
