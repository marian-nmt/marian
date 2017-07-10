// -*- mode: c++; indent-tabs-mode: nil; tab-width: 2 -*-
#pragma once

#include <boost/program_options.hpp>

#include <sys/ioctl.h>
#include <unistd.h>
#include "3rd_party/yaml-cpp/yaml.h"
#include "common/file_stream.h"
#include "common/logging.h"

namespace marian {

// try to determine the width of the terminal
uint16_t guess_terminal_width(uint16_t max_width = 180);

void OutputYaml(const YAML::Node node, YAML::Emitter& out);

class ConfigParser {
public:
  ConfigParser(int argc,
         char** argv,
         bool validate = true,
         bool translate = false,
         bool rescore = false)
      : cmdline_options_("Allowed options", guess_terminal_width()) {
    parseOptions(argc, argv, validate, translate, rescore);
  }

  void parseOptions(
      int argc, char** argv, bool validate, bool translate, bool rescore);

  YAML::Node getConfig() const;

private:
  boost::program_options::options_description cmdline_options_;
  YAML::Node config_;

  bool has(const std::string& key) const;
  template <typename T>
  T get(const std::string& key) const {
    return config_[key].as<T>();
  }

  void addOptionsCommon(boost::program_options::options_description&, bool);
  void addOptionsModel(boost::program_options::options_description&, bool, bool);
  void addOptionsTraining(boost::program_options::options_description&);
  void addOptionsRescore(boost::program_options::options_description&);
  void addOptionsValid(boost::program_options::options_description&);
  void addOptionsTranslate(boost::program_options::options_description&);

  void validateOptions(bool translate = false, bool rescore = false) const;

};
}
