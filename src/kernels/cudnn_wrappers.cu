#include "kernels/cudnn_wrappers.h"

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

CUDNNWrapper::CUDNNWrapper() {
  CUDNN_CALL(cudnnCreate(&cudnnHandle_));
}

CUDNNWrapper::~CUDNNWrapper() {
  cudnnDestroy(cudnnHandle_);
}

void CUDNNWrapper::setCudnnTensor(cudnnTensorDescriptor_t& desc, Tensor x) {
  setCudnnTensor(desc, x->shape());
}

void CUDNNWrapper::setCudnnTensor(
    cudnnTensorDescriptor_t& desc,
    const Shape& shape) {
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

ConvolutionWrapper::ConvolutionWrapper(
    const Shape& kernelShape,
    const Shape& biasShape,
    int hPad,
    int wPad,
    int hStride,
    int wStride) {
  setKernelDescriptor(kernelShape);
  setConvDescriptor(hPad, wPad, hStride, wStride);
  setCudnnTensor(biasDesc_, biasShape);
}

void ConvolutionWrapper::getOutputShape(
    const Shape& xShape,
    Shape& shape) {
  cudnnTensorDescriptor_t xDesc;
  setCudnnTensor(xDesc, xShape);
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

void ConvolutionWrapper::forward(
    Tensor x,
    Tensor kernels,
    Tensor bias,
    Tensor y) {
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
        kernels->data(),
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
        bias->data(),
        &alpha,
        yDesc,
        y->data()));
  cudnnDestroyTensorDescriptor(xDesc);
  cudnnDestroyTensorDescriptor(yDesc);
}

void ConvolutionWrapper::backward(
    Tensor x,
    Tensor xGrad,
    Tensor kernels,
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
        kernels->data(),
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

ConvolutionWrapper::~ConvolutionWrapper() {
  cudnnDestroyConvolutionDescriptor(convDesc_);
  cudnnDestroyFilterDescriptor(kernelDesc_);
  cudnnDestroyTensorDescriptor(biasDesc_);
}

void ConvolutionWrapper::setConvDescriptor(
    int hPad,
    int wPad,
    int hStride,
    int wStride) {
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

void ConvolutionWrapper::setKernelDescriptor(const Shape& shape) {
  ABORT_IF(shape.size() != 4,
            "CUDN requires tensors 4D. Provided {}", shape.toString());
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

}

#endif
