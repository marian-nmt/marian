#pragma once

#include <iostream>

#include "common/shape.h"
#include "tensors/tensor.h"

#ifdef CUDNN

#include <cudnn.h>

namespace marian {

class CUDNNWrapper {
public:
  CUDNNWrapper();
  virtual ~CUDNNWrapper();

protected:
  void setCudnnTensor(cudnnTensorDescriptor_t& desc, Tensor x);

  void setCudnnTensor(cudnnTensorDescriptor_t& desc, const Shape& shape);

protected:
  cudnnHandle_t cudnnHandle_;
};

class ConvolutionWrapper : public CUDNNWrapper {
public:
  ConvolutionWrapper(const Shape& kernelShape,
                     const Shape& biasShape,
                     int hPad = 0,
                     int wPad = 0,
                     int hStride = 1,
                     int wStride = 1);

  void getOutputShape(const Shape& xShape, Shape& shape);

  virtual ~ConvolutionWrapper();

  void forward(Tensor x, Tensor Kernels, Tensor bias, Tensor y);

  void backward(Tensor x,
                Tensor xGrad,
                Tensor kernels,
                Tensor kernelGrad,
                Tensor biasGrad,
                Tensor yGrad);

protected:
  void setConvDescriptor(int hPad, int wPad, int hStride, int wStride);

  void setKernelDescriptor(const Shape& shape);

protected:
  cudnnConvolutionDescriptor_t convDesc_;
  cudnnFilterDescriptor_t kernelDesc_;
  cudnnTensorDescriptor_t biasDesc_;
};

class PoolingWrapper : public CUDNNWrapper {
public:
  PoolingWrapper(int height,
                 int width,
                 int padHeight,
                 int padWidth,
                 int strideHeight,
                 int strideWidth,
                 std::string mode);

  void getOutputShape(const Shape& xShape, Shape& shape);

  void forward(Tensor x, Tensor y);

  void backward(Tensor x, Tensor xGrad, Tensor y, Tensor yGrad);

  virtual ~PoolingWrapper();

protected:
  void setPoolingDescriptor(int height,
                            int width,
                            int padHeight,
                            int padWidth,
                            int strideHeight,
                            int strideWidth);

protected:
  cudnnPoolingDescriptor_t poolingDesc_;
  cudnnPoolingMode_t poolingMode_;
};
}  // namespace marian

#else

namespace marian {

class CUDNNWrapper {
public:
  CUDNNWrapper();
  virtual ~CUDNNWrapper();
};

class ConvolutionWrapper : public CUDNNWrapper {
public:
  ConvolutionWrapper(const Shape& kernelShape,
                     const Shape& biasShape,
                     int hPad = 1,
                     int wPad = 1,
                     int hStride = 1,
                     int wStride = 1);

  void getOutputShape(const Shape& xShape, Shape& shape);

  virtual ~ConvolutionWrapper();

  void forward(Tensor x, Tensor Kernels, Tensor bias, Tensor y);

  void backward(Tensor x,
                Tensor xGrad,
                Tensor kernels,
                Tensor kernelGrad,
                Tensor biasGrad,
                Tensor yGrad);
};

class PoolingWrapper : public CUDNNWrapper {
public:
  PoolingWrapper(int height,
                 int width,
                 int padHeight,
                 int padWidth,
                 int strideHeight,
                 int strideWidth,
                 std::string mode);

  void getOutputShape(const Shape& xShape, Shape& shape);

  void forward(Tensor x, Tensor y);

  void backward(Tensor x, Tensor xGrad, Tensor y, Tensor yGrad);

  virtual ~PoolingWrapper();
};
}  // namespace marian

#endif
