#pragma once

#include <string>
#include <vector>
#include <memory>

namespace amunmt {

class Preprocessor {
  public:
    virtual std::vector<std::string> Preprocess(const std::vector<std::string> input) const = 0;
    virtual ~Preprocessor() {}
};

using PreprocessorPtr = std::unique_ptr<Preprocessor>;

class Postprocessor {
  public:
    virtual std::vector<std::string> Postprocess(const std::vector<std::string> input) const = 0;
    virtual ~Postprocessor() {}
};
using PostprocessorPtr = std::unique_ptr<Postprocessor>;

class Processor : public Preprocessor, public Postprocessor {
  public:
    virtual ~Processor() {}
};
using ProcessorPtr = std::unique_ptr<Processor>;

}
