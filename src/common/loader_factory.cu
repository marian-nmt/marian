#include "loader_factory.h"
#include "scorer.h"
#include "gpu/decoder/encoder_decoder.h"
#include "gpu/decoder/ape_penalty.h"

#ifdef KENLM
#include "gpu/decoder/language_model.h"
#endif

LoaderPtr LoaderFactory::Create(const std::string& name,
						const YAML::Node& config) {
  UTIL_THROW_IF2(!config["type"],
				 "Missing scorer type in config file");

  auto type = config["type"].as<std::string>();
  IF_MATCH_RETURN(type, "Nematus", EncoderDecoderLoader);
  IF_MATCH_RETURN(type, "nematus", EncoderDecoderLoader);
  IF_MATCH_RETURN(type, "NEMATUS", EncoderDecoderLoader);

  IF_MATCH_RETURN(type, "Ape", ApePenaltyLoader);
  IF_MATCH_RETURN(type, "ape", ApePenaltyLoader);
  IF_MATCH_RETURN(type, "APE", ApePenaltyLoader);
#ifdef KENLM
  IF_MATCH_RETURN(type, "KenLM", KenLMLoader)
  IF_MATCH_RETURN(type, "kenlm", KenLMLoader)
  IF_MATCH_RETURN(type, "KENLM", KenLMLoader)
#endif
  UTIL_THROW2("Unknown scorer in config file: " << type);
}
