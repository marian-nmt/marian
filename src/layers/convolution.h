#pragma once

#include "layers/generic.h"
#include "marian.h"

#include <string>

namespace marian {

#ifdef CUDNN
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
  CharConvPooling(const std::string& prefix,
                  int kernelHeight,
                  std::vector<int> kernelWidths,
                  std::vector<int> kernelNums,
                  int stride)
      : name_(prefix),
        size_((int)kernelNums.size()),
        kernelHeight_(kernelHeight),
        kernelWidths_(kernelWidths),
        kernelNums_(kernelNums),
        stride_(stride) {}

  Expr operator()(Expr x, Expr mask) {
    auto graph = x->graph();

    auto masked = x * mask;
    auto xNCHW = convert2cudnnFormat(masked);
    auto maskNCHW = convert2cudnnFormat(mask);

    Expr input = xNCHW;
    std::vector<Expr> outputs;

    for(int i = 0; i < size_; ++i) {
      int kernelWidth = kernelWidths_[i];
      int kernelNum = kernelNums_[i];
      int padWidth = kernelWidth / 2;

      auto output
          = convolution(graph)  //
            ("prefix",
             name_ + "_width_" + std::to_string(kernelWidth))             //
            ("kernel-dims", std::make_pair(kernelWidth, x->shape()[-1]))  //
            ("kernel-num", kernelNum)                                     //
            ("paddings", std::make_pair(padWidth, 0))
                .apply(input);

      auto relued = relu(output);
      auto output2 = pooling_with_masking(
          relued, maskNCHW, stride_, kernelWidth % 2 == 0);

      output2 = reshape(
          output2,
          {output2->shape()[-1], output2->shape()[0], output2->shape()[1]});
      outputs.push_back(output2);
    }

    auto concatenated = concatenate(outputs, -1);

    return concatenated;
  }

protected:
  std::string name_;
  int size_;
  int kernelHeight_;
  std::vector<int> kernelWidths_;
  std::vector<int> kernelNums_;
  int stride_;
};
#endif

}  // namespace marian
