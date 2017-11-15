#pragma once

#include <string>

#include "layers/generic.h"
#include "graph/expression_graph.h"

namespace marian {

class Convolution : public Factory {
protected:
  Ptr<Options> getOptions() { return options_; }

public:
  Convolution(Ptr<ExpressionGraph> graph);

  Expr apply(Expr x);

  virtual Expr apply(const std::vector<Expr>&);
};

typedef Accumulator<Convolution> convolution;


class CharConvPooling {
  public:
    CharConvPooling(
      const std::string& prefix,
      int kernelHeight,
      std::vector<int> kernelWidths,
      std::vector<int> kernelNums)
      : name_(prefix),
        size_(kernelNums.size()),
        kernelHeight_(kernelHeight),
        kernelWidths_(kernelWidths),
        kernelNums_(kernelNums) {}

    Expr operator()(Expr x, Expr mask) {
      auto graph = x->graph();

      auto masked = x * mask;
      auto xNCHW = convert2cudnnFormat(masked);
      auto maskNCHW = convert2cudnnFormat(mask);

      int layerIn = xNCHW->shape()[1];
      Expr input = xNCHW;
      std::vector<Expr> outputs;

      for (int i = 0; i < size_; ++i) {
        int kernelWidth = kernelWidths_[i];
        int kernelNum = kernelNums_[i];
        int padWidth = kernelWidth / 2;


        auto output = convolution(graph)
          ("prefix", name_)
          ("kernel-dims", std::make_pair(kernelWidth, x->shape()[-1]))
          ("kernel-num", kernelNum)
          ("paddings", std::make_pair(padWidth, 0))
          .apply(input);;
        auto relued = relu(output);
        auto output2 = pooling_with_masking(relued, maskNCHW, 5, kernelWidth % 2 == 0);

        outputs.push_back(output2);
      }

      auto concated = concatenate(outputs, 1);

      return concated;
    }

  private:
    std::string name_;
    int size_;

  protected:
    int kernelHeight_;
    std::vector<int> kernelWidths_;
    std::vector<int> kernelNums_;
};

}
