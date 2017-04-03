#pragma once

#include <boost/program_options.hpp>

#include "3rd_party/yaml-cpp/yaml.h"
#include "common/logging.h"
#include "common/file_stream.h"

namespace marian {

class Config {
  public:

    static size_t seed;

    Config(int argc, char** argv, bool validate=true, bool translate=false)
     : cmdline_options_("Allowed options") {
      addOptions(argc, argv, validate, translate);
      log();
    }

    bool has(const std::string& key) const;

    YAML::Node get(const std::string& key) const;

    template <typename T>
    T get(const std::string& key) const {
      return config_[key].as<T>();
    }

    const YAML::Node& get() const;
    YAML::Node& get();

    YAML::Node operator[](const std::string& key) const {
      return get(key);
    }

    void addOptions(int argc, char** argv, bool validate, bool translate);

    void addOptionsCommon(boost::program_options::options_description&);
    void addOptionsModel(boost::program_options::options_description&, bool);
    void addOptionsTraining(boost::program_options::options_description&);
    void addOptionsValid(boost::program_options::options_description&);

    void addOptionsTranslate(boost::program_options::options_description& desc);


    void log();
    void validate(bool translate=false) const;

    void OutputRec(const YAML::Node node, YAML::Emitter& out) const;

    template <class OStream>
    friend OStream& operator<<(OStream& out, const Config& config) {
      YAML::Emitter outYaml;
      config.OutputRec(config.get(), outYaml);
      out << outYaml.c_str();
      return out;
    }

    void save(const std::string& name) {
      OutputFileStream out(name);
      (std::ostream&)out << *this;
    }


  private:
    boost::program_options::options_description cmdline_options_;
    std::string inputPath;
    YAML::Node config_;
};

}
