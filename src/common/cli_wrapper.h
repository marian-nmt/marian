#pragma once

#include "3rd_party/CLI/CLI.hpp"
#include "3rd_party/yaml-cpp/yaml.h"

#include <type_traits>
#include <utility>
#include <map>
#include <string>

#include <iostream>

namespace marian {
namespace cli {

class some {
  using id = size_t;

  template <typename T>
  struct type {
    static void id() {}
  };

  template <typename T>
  static id type_id() {
    return reinterpret_cast<id>(&type<T>::id);
  }

  template <typename T>
  using decay = typename std::decay<T>::type;

  template <typename T>
  using none = typename std::enable_if<!std::is_same<some, T>::value>::type;

  struct base {
    virtual ~base() {}
    virtual bool is(id) const = 0;
    virtual base *copy() const = 0;
  } *p = nullptr;

  template <typename T>
  struct data : base, std::tuple<T> {
    using std::tuple<T>::tuple;

    T &get() & { return std::get<0>(*this); }
    T const &get() const & { return std::get<0>(*this); }

    bool is(id i) const override { return i == type_id<T>(); }
    base *copy() const override { return new data{get()}; }
  };

  template <typename T>
  T &stat() {
    return static_cast<data<T> &>(*p).get();
  }

  template <typename T>
  T const &stat() const {
    return static_cast<data<T> const &>(*p).get();
  }

  template <typename T>
  T &dyn() {
    return dynamic_cast<data<T> &>(*p).get();
  }

  template <typename T>
  T const &dyn() const {
    return dynamic_cast<data<T> const &>(*p).get();
  }

public:
  some() {}
  ~some() { delete p; }

  some(some &&s) : p{s.p} { s.p = nullptr; }
  some(some const &s) : p{s.p->copy()} {}

  template <typename T, typename U = decay<T>, typename = none<U>>
  some(T &&x) : p{new data<U>{std::forward<T>(x)}} {}

  some &operator=(some s) {
    swap(*this, s);
    return *this;
  }

  friend void swap(some &s, some &r) { std::swap(s.p, r.p); }

  void clear() {
    delete p;
    p = nullptr;
  }

  bool empty() const { return p; }

  template <typename T>
  bool is() const {
    return p ? p->is(type_id<T>()) : false;
  }

  template <typename T>
  T &&as() && {
    return std::move(stat<T>());
  }
  template <typename T>
  T &as() & {
    return stat<T>();
  }
  template <typename T>
  T const &as() const & {
    return stat<T>();
  }

  template <typename T>
  T &&cast() && {
    return std::move(dyn<T>());
  }
  template <typename T>
  T &cast() & {
    return dyn<T>();
  }
  template <typename T>
  T const &cast() const & {
    return dyn<T>();
  }

  template <typename T>
  operator T &&() && {
    return std::move(as<T>());
  }
  template <typename T>
  operator T &() & {
    return as<T>();
  }
  template <typename T>
  operator T const &() const & {
    return as<T>();
  }
};

template <typename S, typename T>
S &operator<<(S &s, const std::vector<T> &v) {
  for(auto &x : v)
    s << x << " ";
  return s;
}

class CLIWrapper {
private:
  // Stores option variables
  std::map<std::string, std::shared_ptr<some>> vars_;
  // Stores option objects
  std::map<std::string, std::shared_ptr<CLI::Option>> opts_;
  // Command-line arguments parser
  CLI::App app_;

public:
  CLIWrapper() {}
  virtual ~CLIWrapper() {}

  template <typename T>
  std::shared_ptr<CLI::Option> add(const std::string &key,
                                   const std::string &args,
                                   const std::string &help,
                                   T val = T()) {
    std::cerr << "CLI::add(" << key << ") ";
    vars_.insert(std::make_pair(key, std::shared_ptr<some>(new some(val))));
    opts_.insert(std::make_pair(key,
                                std::shared_ptr<CLI::Option>(app_.add_option(
                                    args, vars_[key]->as<T>(), help))));
    std::cerr << opts_[key]->get_lnames() << std::endl;
    return opts_[key];
  }

  template <typename T>
  std::shared_ptr<CLI::Option> getOption(const std::string &key) {
    std::cerr << "CLI::getOption(" << key << ") .count=" << opts_[key]->count()
              << std::endl;
    return opts_[key];
  }

  template <typename T>
  T get(const std::string &key) {
    std::cerr << "CLI::get(" << key << ") =" << vars_[key]->as<T>()
              << " .count=" << opts_[key]->count()
              << " .bool=" << (bool)(*opts_[key])
              << " .empty=" << opts_[key]->empty() << std::endl;
    return vars_[key]->as<T>();
  }

  bool parse(int argv, char **argc) {
    app_.parse(argv, argc);
  }
};

template <>
std::shared_ptr<CLI::Option> CLIWrapper::add(const std::string &key,
                                             const std::string &args,
                                             const std::string &help,
                                             bool val) {
  std::cerr << "CLI::add(" << key << ") ";
  vars_.insert(std::make_pair(key, std::shared_ptr<some>(new some(false))));
  opts_.insert(std::make_pair(
      key, std::shared_ptr<CLI::Option>(app_.add_flag(args, help))));
  std::cerr << opts_[key]->get_lnames() << std::endl;
  return opts_[key];
}

}  // namespace cli
}  // namespace marian
