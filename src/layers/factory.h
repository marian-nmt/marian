#pragma once

#include "marian.h"

namespace marian {
/**
 * Base class for constructing models or layers.
 * Its main attribute is options which hold the basic characteristics of the model or the layer.
 */
class Factory : public std::enable_shared_from_this<Factory> {
protected:
  Ptr<Options> options_;

public:
  // construct with empty options
  Factory() : options_(New<Options>()) {}
  // construct with options
  Factory(Ptr<Options> options) : Factory() {
    options_->merge(options);
  }
  // construct with one or more individual option parameters
  // Factory("var1", val1, "var2", val2, ...)
  template <typename T, typename... Args>
  Factory(const std::string& key, T value, Args&&... moreArgs) : Factory() {
    setOpts(key, value, std::forward<Args>(moreArgs)...);
  }
  // construct with options and one or more individual option parameters
  // Factory(options, "var1", val1, "var2", val2, ...)
  template <typename... Args>
  Factory(Ptr<Options> options, Args&&... args) : Factory(options) {
    setOpts(std::forward<Args>(args)...);
  }
  Factory(const Factory& factory) = default;

  virtual ~Factory() {}

  std::string asYamlString() { return options_->asYamlString(); }

  // retrieve an option
  // auto val = opt<T>("var");
  template <typename T>
  T opt(const char* const key) { return options_->get<T>(key); }

  template <typename T>
  T opt(const char* const key, T defaultValue) { return options_->get<T>(key, defaultValue); }

  template <typename T>
  T opt(const std::string& key) { return options_->get<T>(key.c_str()); }

  template <typename T>
  T opt(const std::string& key, T defaultValue) { return options_->get<T>(key.c_str(), defaultValue); }

  // set a single option
  // setOpt("var", val);
  template <typename T>
  void setOpt(const std::string& key, T value) { options_->set(key, value); }

  // set one or more options at once
  // setOpts("var1", val1, "var2", val2, ...);
  template <typename T, typename... Args>
  void setOpts(const std::string& key, T value, Args&&... moreArgs) { options_->set(key, value, std::forward<Args>(moreArgs)...); }

  void mergeOpts(Ptr<Options> options) { options_->merge(options); }

  template <class Cast>
  inline Ptr<Cast> as() { return std::dynamic_pointer_cast<Cast>(shared_from_this()); }

  // @TODO: this fails with 'target type must be a pointer or reference to a defined class'
  //template <class Cast>
  //inline bool is() { return dynamic_cast<Cast>(this) != nullptr; }
  template <class Cast>
  inline bool is() { return std::dynamic_pointer_cast<Cast>(shared_from_this()) != nullptr; }
};
/** Simplest form of Factory that just passes on options to the constructor of a layer. */
template<class Class>
struct ConstructingFactory : public Factory {
  using Factory::Factory;

  Ptr<Class> construct(Ptr<ExpressionGraph> graph) {
    return New<Class>(graph, options_);
  }
};

/**
 * Accumulator<Factory> pattern offers a shortcut to construct models or layers.
 * The options can be passed by a pair of parentheses. E.g., to construct a fully-connected layer:
 * \code{.cpp}
 * auto hidden = mlp::dense()
      ("prefix", "hidden_layer")               // layer name
      ("dim", outDim)                          // output dimension
      ("activation", (int)mlp::act::sigmoid)   // activation function
      .construct(graph);                       // construct this layer in graph
   \endcode
 */
template <class BaseFactory> // where BaseFactory : Factory
class Accumulator : public BaseFactory {
  typedef BaseFactory Factory;

public:
  Accumulator() : Factory() {}
  Accumulator(Ptr<Options> options) : Factory(options) {}
  template <typename... Args>
  Accumulator(Ptr<Options> options, Args&&... moreArgs) : Factory(options, std::forward<Args>(moreArgs)...) {}
  template <typename T, typename... Args>
  Accumulator(const std::string& key, T value, Args&&... moreArgs) : Factory(key, value, std::forward<Args>(moreArgs)...) {}
  Accumulator(const Factory& factory) : Factory(factory) {}
  Accumulator(const Accumulator&) = default;
  Accumulator(Accumulator&&) = default;

  // deprecated chaining syntax
  template <typename T>
  Accumulator& operator()(const std::string& key, T value) {
    Factory::setOpt(key, value);
    return *this;
  }

  Accumulator& operator()(Ptr<Options> options) {
    Factory::mergeOpts(options);
    return *this;
  }

  Accumulator<Factory> clone() {
    return Accumulator<Factory>(Factory::clone());
  }
};
}  // namespace marian
