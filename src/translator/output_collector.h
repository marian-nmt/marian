#pragma once

#include "common/definitions.h"
#include "common/file_stream.h"

#include <mutex>
#include <iostream>
#include <map>

namespace marian {

class PrintingStrategy {
public:
  virtual ~PrintingStrategy() {}
  virtual bool shouldBePrinted(long) = 0;
};

class DefaultPrinting : public PrintingStrategy {
public:
  bool shouldBePrinted(long) override { return true; }
};

class QuietPrinting : public PrintingStrategy {
public:
  bool shouldBePrinted(long) override { return false; }
};

class GeometricPrinting : public PrintingStrategy {
public:
  bool shouldBePrinted(long id) override {
    if(id == 0)
      next_ = (long)start_;
    if(id <= 5)
      return true;
    if(next_ == id) {
      next_ += next_;
      return true;
    }
    return false;
  }

private:
  size_t start_{10};
  long next_{10};
};

class OutputCollector {
public:
  OutputCollector();
  OutputCollector(std::string outFile);

  template <class T>
  OutputCollector(T&& arg) : nextId_(0), outStrm_(new io::OutputFileStream(arg)) {}

  OutputCollector(const OutputCollector&) = delete;

  void Write(long sourceId,
             const std::string& best1,
             const std::string& bestn,
             bool nbest);

  void setPrintingStrategy(Ptr<PrintingStrategy> strategy) {
    printing_ = strategy;
  }

protected:
  typedef std::map<long, std::pair<std::string, std::string>> Outputs;
  Outputs outputs_;
  long nextId_;
  UPtr<std::ostream> outStrm_;
  Ptr<PrintingStrategy> printing_;
  std::mutex mutex_;
};

class StringCollector {
public:
  StringCollector(bool quiet = false);
  StringCollector(const StringCollector&) = delete;

  void add(long sourceId, const std::string& best1, const std::string& bestn);
  std::vector<std::string> collect(bool nbest);

protected:
  long maxId_;  // the largest index of the translated source sentences
  bool quiet_;  // if true do not log best translations
  std::mutex mutex_;

  typedef std::map<long, std::pair<std::string, std::string>> Outputs;
  Outputs outputs_;
};
}  // namespace marian
