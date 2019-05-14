#pragma once

#include "marian.h"

namespace marian {

class Factory : public std::enable_shared_from_this<Factory> {
protected:
  Ptr<Options> options_;

public:
  Factory() : options_(New<Options>()) {}
  Factory(Ptr<Options> options) : Factory() {
    options_->merge(options);
  }

  virtual ~Factory() {}

  Ptr<Options> getOptions() { return options_; }

  std::string str() { return options_->str(); }

  template <typename T>
  T opt(const std::string& key) {
    return options_->get<T>(key);
  }

  template <typename T>
  T opt(const std::string& key, T defaultValue) {
    return options_->get<T>(key, defaultValue);
  }

  template <typename Cast>
  inline Ptr<Cast> as() {
    return std::dynamic_pointer_cast<Cast>(shared_from_this());
  }

  template <typename Cast>
  inline bool is() {
    return as<Cast>() != nullptr;
  }
};

// simplest form of Factory that just passes on options to the constructor of a layer type
template<class Class>
struct ConstructingFactory : public Factory {
  Ptr<Class> construct(Ptr<ExpressionGraph> graph) {
    return New<Class>(graph, options_);
  }
};

template <class BaseFactory>
class Accumulator : public BaseFactory {
  typedef BaseFactory Factory;

public:
  Accumulator() : Factory() {}
  Accumulator(const Factory& factory) : Factory(factory) {}
  Accumulator(const Accumulator&) = default;
  Accumulator(Accumulator&&) = default;

  template <typename T>
  Accumulator& operator()(const std::string& key, T value) {
    Factory::getOptions()->set(key, value);
    return *this;
  }

  Accumulator& operator()(const std::string& yaml) {
    Factory::getOptions()->parse(yaml);
    return *this;
  }

#if 0
  Accumulator& operator()(Config::YamlNode yaml) {
    Factory::getOptions()->merge(yaml);
    return *this;
  }
#endif

  Accumulator& operator()(Ptr<Options> options) {
    Factory::getOptions()->merge(options);
    return *this;
  }

  Accumulator<Factory> clone() {
    return Accumulator<Factory>(Factory::clone());
  }
};
}  // namespace marian
