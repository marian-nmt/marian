#include "loader_factory.h"
#include "cpu/decoder/encoder_decoder.h"

LoaderPtr LoaderFactory::CreateCPU(const std::string& name,
						const YAML::Node& config) {
  UTIL_THROW_IF2(!config["type"],
				 "Missing scorer type in config file");
  auto type = config["type"].as<std::string>();

  IF_MATCH_RETURN(type, "Nematus.CPU", CPU::EncoderDecoderLoader);

  UTIL_THROW2("Unknown scorer in config file: " << type);
}
