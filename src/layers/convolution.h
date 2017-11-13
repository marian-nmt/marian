#pragma once

#include <string>

#include "layers/generic.h"

namespace marian {

class ConvPoolingBase {
  public:
    ConvPoolingBase(const std::string& name)
      : name_(name)
    {}

  protected:
    virtual Expr convert2Marian(Expr x, Expr originalX) {
      std::vector<size_t> newIndeces;
      int batchDim = x->shape()[0];
      int sentenceDim = x->shape()[2];

      auto pooled = reshape(x, {batchDim * sentenceDim, x->shape()[3], 1, x->shape()[1]});

      for (int t = 0; t < sentenceDim; ++t) {
        for (int b = 0; b < batchDim; ++b) {
          newIndeces.push_back(b * sentenceDim + t);
        }
      }

      return reshape(rows(pooled, newIndeces), originalX->shape());
    }

    virtual Expr convert2NCHW(Expr x) {
      std::vector<size_t> newIndeces;
      int batchDim = x->shape()[0];
      int sentenceDim = x->shape()[2];

      for (int b = 0; b < batchDim; ++b) {
        for (int t = 0; t < sentenceDim; ++t) {
          newIndeces.push_back((t * batchDim) + b);
        }
      }

      Shape shape({batchDim, 1, sentenceDim, x->shape()[1]});
      return  reshape(rows(x, newIndeces), shape);
    }

    virtual Expr operator()(Expr x, Expr mask) = 0;

  protected:
    std::string name_;

};

class Convolution : public ConvPoolingBase {
  public:
    Convolution(
      const std::string& name,
      int kernelHeight = 3,
      int kernelWidth = 3,
      int kernelNum = 1,
      int paddingHeight = 0,
      int paddingWidth = 0,
      int strideHeight = 1,
      int strideWidth = 1)
      : ConvPoolingBase(name),
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

      auto kernel = graph->param(name_,
          {layerIn, kernelNum_, kernelHeight_, kernelWidth_},
          keywords::init=inits::glorot_uniform);
      auto bias = graph->param(name_ + "_bias",  {1, kernelNum_, 1, 1},
                               keywords::init=inits::zeros);

      auto output = convolution(x, kernel, bias,
                                paddingHeight_, paddingWidth_,
                                strideHeight_, strideWidth_);

      return output;
    }

    Expr operator()(Expr x, Expr mask) {
      return this->operator()(x, mask, 1);
    }

    Expr operator()(Expr x, Expr mask, int n) {
      auto graph = x->graph();

      auto masked = x * mask;
      auto xNCHW = convert2NCHW(masked);
      auto maskNCHW = convert2NCHW(mask);

      int layerIn = xNCHW->shape()[1];

      Expr input = xNCHW;
      for (int i = 0; i < n; ++i) {
        auto kernel = graph->param(name_ + std::to_string(i),
            {layerIn, kernelNum_, kernelHeight_, kernelWidth_},
            keywords::init=inits::glorot_uniform);
        auto bias = graph->param(name_ + std::to_string(i) + "_bias",  {1, kernelNum_, 1, 1},
                                keywords::init=inits::zeros);

        auto output = convolution(input, kernel, bias,
            paddingHeight_, paddingWidth_,
            strideHeight_, strideWidth_);
        input = tanh(input + output) * maskNCHW;
      }

      return convert2Marian(input, x);
    }

  private:
    std::string name_;

  protected:
    int depth_;
    int kernelHeight_;
    int kernelWidth_;
    int kernelNum_;
    int strideHeight_;
    int strideWidth_;
    int paddingHeight_;
    int paddingWidth_;
};


class Pooling : public ConvPoolingBase {
public:
  Pooling(
      const std::string name,
      const std::string type,
      int height = 1,
      int width = 1,
      int paddingHeight = 0,
      int paddingWidth = 0,
      int strideHeight = 1,
      int strideWidth = 1)
    : ConvPoolingBase(name),
      type_(type),
      height_(height),
      width_(width),
      paddingHeight_(paddingHeight),
      paddingWidth_(paddingWidth),
      strideHeight_(strideHeight),
      strideWidth_(strideWidth)
  {
  }

    Expr operator()(Expr x, Expr mask) {
      params_ = {};

      auto masked = x * mask;

      auto xNCHW = convert2NCHW(masked);

      Expr output;
      if (type_ == "max_pooling") {
        output = max_pooling(xNCHW, height_, width_,
                             paddingHeight_, paddingWidth_,
                             strideHeight_, strideWidth_);
      } else if (type_ == "avg_pooling") {
        output = avg_pooling(xNCHW, height_, width_,
                             paddingHeight_, paddingWidth_,
                             strideHeight_, strideWidth_);
      }

      return convert2Marian(output, x) * mask;
    }

  private:
    std::vector<Expr> params_;
    std::string name_;
    std::string type_;

  protected:
    int height_;
    int width_;
    int paddingHeight_;
    int paddingWidth_;
    int strideHeight_;
    int strideWidth_;

};


class CharConvPooling : public ConvPoolingBase {
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
