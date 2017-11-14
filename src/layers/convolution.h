#pragma once

#include <string>

#include "layers/generic.h"

namespace marian {

class Convolution {
  public:
    Convolution(
      const std::string& prefix,
      int kernelHeight = 3,
      int kernelWidth = 3,
      int kernelNum = 1,
      int paddingHeight = 0,
      int paddingWidth = 0,
      int strideHeight = 1,
      int strideWidth = 1)
      : prefix_(prefix),
        kernelHeight_(kernelHeight),
        kernelWidth_(kernelWidth),
        kernelNum_(kernelNum),
        strideHeight_(strideHeight),
        strideWidth_(strideWidth),
        paddingHeight_(paddingHeight),
        paddingWidth_(paddingWidth)
    {
    }

    Expr operator()(Expr x) {
      auto graph = x->graph();

      int layerIn = x->shape()[1];

      auto kernel = graph->param(prefix_ + "_conv_kernels",
          {layerIn, kernelNum_, kernelHeight_, kernelWidth_},
          keywords::init=inits::glorot_uniform);
      auto bias = graph->param(prefix_ + "_conv_bias",  {1, kernelNum_, 1, 1},
                               keywords::init=inits::zeros);

      auto output = convolution(x, kernel, bias,
                                paddingHeight_,
                                paddingWidth_,
                                strideHeight_,
                                strideWidth_);

      return output;
    }

  protected:
    std::string prefix_;
    int depth_;
    int kernelHeight_;
    int kernelWidth_;
    int kernelNum_;
    int strideHeight_;
    int strideWidth_;
    int paddingHeight_;
    int paddingWidth_;
};


class CharConvPooling {
  public:
    CharConvPooling(
      const std::string& name,
      int kernelHeight,
      std::vector<int> kernelWidths,
      std::vector<int> kernelNums)
      : ConvPoolingBase(name),
        size_(kernelNums.size()),
        kernelHeight_(kernelHeight),
        kernelWidths_(kernelWidths),
        kernelNums_(kernelNums) {}

    Expr operator()(Expr x, Expr mask) {
      auto graph = x->graph();

      auto masked = x * mask;
      auto xNCHW = convert2NCHW(masked);
      auto maskNCHW = convert2NCHW(mask);

      int layerIn = xNCHW->shape()[1];
      Expr input = xNCHW;
      std::vector<Expr> outputs;

      for (int i = 0; i < size_; ++i) {
        int kernelWidth = kernelWidths_[i];
        int kernelDim = kernelNums_[i];
        int padWidth = kernelWidth / 2;

        auto kernel = graph->param(name_ + std::to_string(i),
            {layerIn, kernelDim, kernelWidth, x->shape()[1]},
             keywords::init=inits::glorot_uniform);
        auto bias = graph->param(name_ + std::to_string(i) + "_bias",  {1, kernelDim, 1, 1},
                                 keywords::init=inits::zeros);

        auto output = convolution(input, kernel, bias, padWidth, 0, 1, 1);
        auto relued = relu(output);
        auto output2 = max_pooling2(relued, maskNCHW, 5, kernelWidth % 2 == 0);

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
