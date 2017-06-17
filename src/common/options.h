#pragma once

#include <string>
#include <sstream>
#include "common/definitions.h"
#include "3rd_party/yaml-cpp/yaml.h"

namespace marian {

class Options {
  protected:
    YAML::Node options_;

  public:
    YAML::Node& getOptions() {
      return options_;
    }

    void parse(const std::string& yaml) {
      auto node = YAML::Load(yaml);
      for(auto it : node)
        options_[it.first.as<std::string>()] = it.second;
    }

    void merge(YAML::Node& node) {
      for(auto it : node)
        options_[it.first.as<std::string>()] = it.second;
    }

    std::string str() {
      std::stringstream ss;
      ss << options_;
      return ss.str();
    }

    template <typename T>
    void set(const std::string& key, T value) {
      options_[key] = value;
    }

    template <typename T>
    T get(const std::string& key) {
      UTIL_THROW_IF2(!has(key), "Required option \"" << key << "\" has not been set");
      return options_[key].as<T>();
    }

    template <typename T>
    T get(const std::string& key, T defaultValue) {
      if(has(key))
        return options_[key].as<T>();
      else
        return defaultValue;
    }

    bool has(const std::string& key) const {
      return options_[key];
    }
};

template <class Obj>
struct DefaultCreate {
  template <typename ...Args>
  static Ptr<Obj> create(Ptr<ExpressionGraph> graph, Ptr<Options> options, Args ...args) {
    return New<Obj>(graph, options, args...);
  }
};

template <class Obj, class Create=DefaultCreate<Obj>>
class Builder {
  protected:
    Ptr<Options> options_;
    Ptr<ExpressionGraph> graph_;

  public:
    Builder(Ptr<ExpressionGraph> graph)
    : options_(New<Options>()), graph_(graph) {}

    Ptr<Options> getOptions() {
      return options_;
    }

    virtual std::string str() {
      return options_->str();
    }

    template <typename T>
    Builder& operator()(const std::string& key, T value) {
      options_->set(key, value);
      return *this;
    }

    Builder& operator()(const std::string& yaml) {
      options_->parse(yaml);
      return *this;
    }

    Builder& operator()(YAML::Node yaml) {
      options_->merge(yaml);
      return *this;
    }

    template <typename T>
    T get(const std::string& key) {
      return options_->get<T>(key);
    }

    template <typename ...Args>
    Ptr<Obj> create(Args ...args) {
      return Create::create(graph_, options_, args...);
    }

};

}