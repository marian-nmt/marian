#pragma once

#include <vector>
#include <yaml-cpp/yaml.h>

#include "exception.h"
#include "scorer.h"
#include "encoder_decoder.h"
#include "ape_penalty.h"

#ifdef KENLM
#include "language_model.h"
#endif

#define IF_MATCH_RETURN(typeVar, typeStr, LoaderType) \
do { \
  if(typeVar == typeStr) { \
    LoaderPtr loader(new LoaderType(name, config)); \
    loader->Load(); \
    return loader; \
  } \
} while(0)

class LoaderFactory {
  public:
    static LoaderPtr Create(const std::string& name,
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
};

