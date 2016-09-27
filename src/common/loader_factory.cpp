#include "loader_factory.h"
#include "cpu/decoder/encoder_decoder.h"

#ifdef NO_CUDA
LoaderPtr LoaderFactory::Create(const std::string& name,
						const YAML::Node& config)
{
	Loader *loader;
	loader = CreateCPU(name, config);
	if (loader) {
		return LoaderPtr(loader);
	}

	std::string type = config["type"].as<std::string>();
	UTIL_THROW2("Unknown scorer in config file: " << type);
}
#endif


Loader *LoaderFactory::CreateCPU(const std::string& name,
						const YAML::Node& config) {
  UTIL_THROW_IF2(!config["type"],
				 "Missing scorer type in config file");
  std::string type = config["type"].as<std::string>();

  IF_MATCH_RETURN(type, "Nematus.CPU", CPU::EncoderDecoderLoader);

  return NULL;
}
