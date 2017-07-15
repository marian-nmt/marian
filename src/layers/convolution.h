#pragma once

#include <string>

#include "layers/generic.h"

namespace marian {

class Convolution {
  public:
    Convolution(
        const std::string& name,
        int kernelHeight = 3,
        int kernelWidth = 3,
        int kernelNum = 1,
        int depth = 1)
      : name_(name),
        depth_(depth),
        kernelHeight_(kernelHeight),
        kernelWidth_(kernelWidth),
        kernelNum_(kernelNum) {
    }

    Expr operator()(Expr x) {
      params_.clear();
      auto graph = x->graph();

      int layerIn = x->shape()[1];

      auto kernel = graph->param(name_ + "_kernels",
          {layerIn, kernelNum_, kernelHeight_, kernelWidth_},
          keywords::init=inits::glorot_uniform);
      auto bias = graph->param(name_ + "_bias",  {1, kernelNum_, 1, 1},
                               keywords::init=inits::zeros);
      params_.push_back(kernel);
      params_.push_back(bias);

      auto output = convolution(x, kernel, bias);

      return output;
    }

    Expr operator()(Expr x, Expr mask) {
      params_ = {};

      auto graph = x->graph();

      std::vector<size_t> newIndeces;
      int batchDim = x->shape()[0];
      int sentenceDim = x->shape()[2];

      for (int b = 0; b < batchDim; ++b) {
        for (int t = 0; t < sentenceDim; ++t) {
          newIndeces.push_back((t * batchDim) + b);
        }
      }

      auto masked = reshape(x * mask,
                            {batchDim * sentenceDim, x->shape()[1], 1, x->shape()[3]});
      auto shuffled_X = reshape(rows(masked, newIndeces),
                                     {batchDim, 1, sentenceDim, x->shape()[1]});
      auto shuffled_mask = reshape(rows(mask, newIndeces),
                                   {batchDim, 1, sentenceDim, mask->shape()[1]});


      Expr previousInput = shuffled_X;

      std::string kernel_name = name_ + "kernels";
      auto kernel = graph->param(kernel_name,  {kernelNum_, kernelHeight_, kernelWidth_},
                                  keywords::init=inits::glorot_uniform);
      auto bias = graph->param(name_ + "_bias",  {1, kernelNum_, 1, 1},
                                 keywords::init=inits::zeros);
      params_.push_back(kernel);
      params_.push_back(bias);

      auto input = previousInput * shuffled_mask;
      previousInput = convolution(input, kernel, bias);

      auto reshapedOutput = reshape(previousInput, {previousInput->shape()[0] * previousInput->shape()[2],
                                                    previousInput->shape()[1], 1, x->shape()[3]});

      newIndeces.clear();
      for (int t = 0; t < sentenceDim; ++t) {
        for (int b = 0; b < batchDim; ++b) {
          newIndeces.push_back(b * previousInput->shape()[2] + t);
        }
      }

      auto reshaped = reshape(rows(reshapedOutput, newIndeces),
                         {x->shape()[0], previousInput->shape()[1], x->shape()[2], x->shape()[3]});
      // debug(reshaped, "RESHAPED");
      return reshaped * mask;
    }

  private:
    std::vector<Expr> params_;
    std::string name_;

  protected:
    int depth_;
    int kernelHeight_;
    int kernelWidth_;
    int kernelNum_;
};

class MaxPooling {
  public:
    MaxPooling(
        const std::string& name,
        int height = 1,
        int width = 1,
        int strideHeight = 1,
        int strideWidth = 1)
      : name_(name),
        height_(height),
        width_(width),
        strideHeight_(strideHeight),
        strideWidth_(strideWidth) {
    }

    Expr operator()(Expr x) {
      params_ = {};
      return max_pooling(x, height_, width_, 0, 0, strideHeight_, strideWidth_);
    }

  private:
    std::vector<Expr> params_;
    std::string name_;

  protected:
    int height_;
    int width_;
    int strideHeight_;
    int strideWidth_;

};


// class Pooling : public Layer {
  // public:
    // Pooling(const std::string& name)
      // : Layer(name) {
    // }

    // Expr operator()(Expr x, Expr xMask) {
      // params_ = {};

      // std::vector<size_t> newIndeces;
      // int batchDim = x->shape()[0];
      // int sentenceDim = x->shape()[2];

      // for (int b = 0; b < batchDim; ++b) {
        // for (int t = 0; t < sentenceDim; ++t) {
          // newIndeces.push_back((t * batchDim) + b);
        // }
      // }

      // auto masked = reshape(x * xMask, {batchDim * sentenceDim, x->shape()[1], 1, x->shape()[3]});
      // auto newX = reshape(rows(masked, newIndeces), {batchDim, x->shape()[1], sentenceDim, 1});

      // auto pooled = reshape(avg_pooling(newX), {batchDim * sentenceDim, x->shape()[1], 1, x->shape()[3]});

      // newIndeces.clear();
      // for (int t = 0; t < sentenceDim; ++t) {
        // for (int b = 0; b < batchDim; ++b) {
          // newIndeces.push_back(b * sentenceDim + t);
        // }
      // }

      // auto reshaped = reshape(rows(pooled, newIndeces), x->shape());
      // return reshaped * xMask;
    // }
// };

}
