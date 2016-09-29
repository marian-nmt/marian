#pragma once

#include <string>
#include <vector>

class Processor {
  public:
    virtual std::vector<std::string> Preprocess(const std::vector<std::string> input) = 0;

    virtual std::vector<std::string> Postprocess(const std::vector<std::string> input) = 0;
};
