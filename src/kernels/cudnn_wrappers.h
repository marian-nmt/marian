#pragma once

#ifdef CUDNN

#include <cudnn.h>

#define CUDNN_CALL(x)                 \
  do {                                \
    if((x) != CUDNN_STATUS_SUCCESS) { \
      printf("Error (%s) at %s:%d\n", \
             cudnnGetErrorString(x),  \
             __FILE__,                \
             __LINE__);               \
    }                                 \
  } while(0)

namespace marian {

class CUDNNWrapper {
public:
  CUDNNWrapper() {
    CUDNN_CALL(cudnnCreate(&cudnnHandle_));
  }

  virtual ~CUDNNWrapper() {
    cudnnDestroy(cudnnHandle_);
  }

protected:
  void setCudnnTensor(cudnnTensorDescriptor_t& desc, Tensor x) {
    setCudnnTensor(desc, x->shape());
  }

  void setCudnnTensor(cudnnTensorDescriptor_t& desc, const Shape& shape) {
    CUDNN_CALL(cudnnCreateTensorDescriptor(&desc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(
          desc,
          CUDNN_TENSOR_NCHW,
          CUDNN_DATA_FLOAT,
          shape[0],
          shape[1],
          shape[2],
          shape[3]));
  }

protected:
  cudnnHandle_t cudnnHandle_;
};

class ConvolutionWrapper : public CUDNNWrapper {
public:
  ConvolutionWrapper(
      Tensor kernels,
      Tensor bias,
      int hPad = 1,
      int wPad = 1,
      int hStride = 1,
      int wStride = 1) {
    setKernelDescriptor(kernels->shape());
    setConvDescriptor(hPad, wPad, hStride, wStride);
    setCudnnTensor(biasDesc_, bias);
  }

  ~ConvolutionWrapper() {
    cudnnDestroyConvolutionDescriptor(convDesc_);
    cudnnDestroyFilterDescriptor(kernelDesc_);
    cudnnDestroyTensorDescriptor(biasDesc_);
  }

  void getOutputShape(Tensor x, Shape& shape) {
    cudnnTensorDescriptor_t xDesc;
    setCudnnTensor(xDesc, x);
    shape.resize(4);
    CUDNN_CALL(cudnnGetConvolution2dForwardOutputDim(
          convDesc_,
          xDesc,
          kernelDesc_,
          shape.data(),
          shape.data() + 1,
          shape.data() + 2,
          shape.data() + 3));
    cudnnDestroyTensorDescriptor(xDesc);
  }

  void forward(Tensor x, Tensor y) {
    cudnnTensorDescriptor_t xDesc, yDesc;
    setCudnnTensor(xDesc, x);
    setCudnnTensor(yDesc, y);

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cudaSetDevice(x->getDevice());

    CUDNN_CALL(cudnnConvolutionForward(
          cudnnHandle_,
          &alpha,
          xDesc,
          x->data(),
          kernelDesc_,
          kernels_->data(),
          convDesc_,
          CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
          nullptr,
          0,
          &beta,
          yDesc,
          y->data()));
    CUDNN_CALL(cudnnAddTensor(
          cudnnHandle_,
          &alpha,
          biasDesc_,
          bias_->data(),
          &alpha,
          yDesc,
          y->data()));
    cudnnDestroyTensorDescriptor(xDesc);
    cudnnDestroyTensorDescriptor(yDesc);
  }

  void backward(
      Tensor x,
      Tensor xGrad,
      Tensor kernelGrad,
      Tensor biasGrad,
      Tensor yGrad) {
    cudaSetDevice(xGrad->getDevice());

    cudnnTensorDescriptor_t xDesc, yDesc;
    setCudnnTensor(xDesc, xGrad);
    setCudnnTensor(yDesc, yGrad);

    const float alpha = 1.0f;
    const float beta = 1.0f;

    CUDNN_CALL(cudnnConvolutionBackwardData(
          cudnnHandle_,
          &alpha,
          kernelDesc_,
          kernels_->data(),
          yDesc,
          yGrad->data(),
          convDesc_,
          CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
          nullptr,
          0,
          &beta,
          xDesc,
          xGrad->data()));

    CUDNN_CALL(cudnnConvolutionBackwardFilter(
        cudnnHandle_,
        &alpha,
        xDesc,
        x->data(),
        yDesc,
        yGrad->data(),
        convDesc_,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
        nullptr,
        0,
        &beta,
        kernelDesc_,
        kernelGrad->data()));

    CUDNN_CALL(cudnnConvolutionBackwardBias(
          cudnnHandle_,
          &alpha,
          yDesc,
          yGrad->data(),
          &beta,
          biasDesc_,
          biasGrad->data()));

    cudnnDestroyTensorDescriptor(xDesc);
    cudnnDestroyTensorDescriptor(yDesc);
  }

protected:
  void setConvDescriptor(int hPad, int wPad, int hStride, int wStride) {
    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&convDesc_));

    CUDNN_CALL(cudnnSetConvolution2dDescriptor(
          convDesc_,
          hPad,
          wPad,
          hStride,
          wStride,
          1,
          1,  // upscales
#if CUDNN_MAJOR > 5
          CUDNN_CROSS_CORRELATION,
          CUDNN_DATA_FLOAT));
#else
          CUDNN_CROSS_CORRELATION));
#endif
  }

  void setKernelDescriptor(const Shape& shape) {
    CUDNN_CALL(cudnnCreateFilterDescriptor(&kernelDesc_));

    int layerIn = shape[0];
    int layerOut = shape[1];
    int kernelH = shape[2];
    int kernelW = shape[3];

    CUDNN_CALL(cudnnSetFilter4dDescriptor(
          kernelDesc_,
          CUDNN_DATA_FLOAT,
          CUDNN_TENSOR_NCHW,
          layerOut,
          layerIn,
          kernelH,
          kernelW));
  }

protected:
  Tensor kernels_;
  Tensor bias_;
  cudnnConvolutionDescriptor_t convDesc_;
  cudnnFilterDescriptor_t kernelDesc_;
  cudnnTensorDescriptor_t biasDesc_;

};

#endif

}

