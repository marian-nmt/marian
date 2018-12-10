#include "tensors/gpu/cudnn_wrappers.h"

namespace marian {

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

CUDNNWrapper::CUDNNWrapper() {
  CUDNN_CALL(cudnnCreate(&cudnnHandle_));
}

CUDNNWrapper::~CUDNNWrapper() {
  // std::cerr << "destroy wrapper" << std::endl;
  CUDNN_CALL(cudnnDestroy(cudnnHandle_));
}

void CUDNNWrapper::setCudnnTensor(cudnnTensorDescriptor_t& desc, Tensor x) {
  setCudnnTensor(desc, x->shape());
}

void CUDNNWrapper::setCudnnTensor(cudnnTensorDescriptor_t& desc,
                                  const Shape& shape) {
  CUDNN_CALL(cudnnCreateTensorDescriptor(&desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(desc,
                                        CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT,
                                        shape[0],
                                        shape[1],
                                        shape[2],
                                        shape[3]));
}

/******************************************************************************
 * ConvolutionWrapper
 *****************************************************************************/

ConvolutionWrapper::ConvolutionWrapper(const Shape& kernelShape,
                                       const Shape& biasShape,
                                       int hPad,
                                       int wPad,
                                       int hStride,
                                       int wStride) {
  setKernelDescriptor(kernelShape);
  setConvDescriptor(hPad, wPad, hStride, wStride);
  setCudnnTensor(biasDesc_, biasShape);
}

void ConvolutionWrapper::getOutputShape(const Shape& xShape, Shape& shape) {
  cudnnTensorDescriptor_t xDesc;
  setCudnnTensor(xDesc, xShape);
  shape.resize(4);
  CUDNN_CALL(cudnnGetConvolution2dForwardOutputDim(convDesc_,
                                                   xDesc,
                                                   kernelDesc_,
                                                   shape.data(),
                                                   shape.data() + 1,
                                                   shape.data() + 2,
                                                   shape.data() + 3));
  cudnnDestroyTensorDescriptor(xDesc);
}

void ConvolutionWrapper::forward(Tensor x,
                                 Tensor kernels,
                                 Tensor bias,
                                 Tensor y) {
  cudaSetDevice(x->getDeviceId().no);

  cudnnTensorDescriptor_t xDesc, yDesc;
  setCudnnTensor(xDesc, x);
  setCudnnTensor(yDesc, y);

  const float alpha = 1.0f;
  const float beta = 0.0f;

  CUDNN_CALL(cudnnConvolutionForward(cudnnHandle_,
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
      cudnnHandle_, &alpha, biasDesc_, bias->data(), &alpha, yDesc, y->data()));
  cudnnDestroyTensorDescriptor(xDesc);
  cudnnDestroyTensorDescriptor(yDesc);
}

void ConvolutionWrapper::backward(Tensor x,
                                  Tensor xGrad,
                                  Tensor kernels,
                                  Tensor kernelGrad,
                                  Tensor biasGrad,
                                  Tensor yGrad) {
  cudaSetDevice(xGrad->getDeviceId().no);

  cudnnTensorDescriptor_t xDesc, yDesc;
  setCudnnTensor(xDesc, xGrad);
  setCudnnTensor(yDesc, yGrad);

  const float alpha = 1.0f;
  const float beta = 1.0f;

  CUDNN_CALL(cudnnConvolutionBackwardData(cudnnHandle_,
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

  CUDNN_CALL(cudnnConvolutionBackwardFilter(cudnnHandle_,
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

  CUDNN_CALL(cudnnConvolutionBackwardBias(cudnnHandle_,
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
  // std::cerr << "destroy conv-wrapper" << std::endl;
  cudnnDestroyConvolutionDescriptor(convDesc_);
  cudnnDestroyFilterDescriptor(kernelDesc_);
  cudnnDestroyTensorDescriptor(biasDesc_);
}

void ConvolutionWrapper::setConvDescriptor(int hPad,
                                           int wPad,
                                           int hStride,
                                           int wStride) {
  CUDNN_CALL(cudnnCreateConvolutionDescriptor(&convDesc_));

#if CUDNN_MAJOR > 5
  CUDNN_CALL(cudnnSetConvolution2dDescriptor(convDesc_,
                                             hPad,
                                             wPad,
                                             hStride,
                                             wStride,
                                             1,
                                             1,  // upscales
                                             CUDNN_CROSS_CORRELATION,
                                             CUDNN_DATA_FLOAT));
#else
  CUDNN_CALL(cudnnSetConvolution2dDescriptor(convDesc_,
                                             hPad,
                                             wPad,
                                             hStride,
                                             wStride,
                                             1,
                                             1,  // upscales
                                             CUDNN_CROSS_CORRELATION));
#endif
}

void ConvolutionWrapper::setKernelDescriptor(const Shape& shape) {
  ABORT_IF(shape.size() != 4,
           "CUDN requires tensors 4D. Provided {}",
           shape.toString());
  CUDNN_CALL(cudnnCreateFilterDescriptor(&kernelDesc_));

  int layerIn = shape[0];
  int layerOut = shape[1];
  int kernelH = shape[2];
  int kernelW = shape[3];

  CUDNN_CALL(cudnnSetFilter4dDescriptor(kernelDesc_,
                                        CUDNN_DATA_FLOAT,
                                        CUDNN_TENSOR_NCHW,
                                        layerOut,
                                        layerIn,
                                        kernelH,
                                        kernelW));
}

/******************************************************************************
 * PoolingWrapper
 *****************************************************************************/

PoolingWrapper::PoolingWrapper(int height,
                               int width,
                               int padHeight,
                               int padWidth,
                               int strideHeight,
                               int strideWidth,
                               std::string mode) {
  if(mode == "max") {
    poolingMode_ = CUDNN_POOLING_MAX;
  } else if(mode == "avg") {
    poolingMode_ = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
  } else {
    ABORT("Unknown pooling mode.");
  }

  setPoolingDescriptor(
      height, width, padHeight, padWidth, strideHeight, strideWidth);
}

void PoolingWrapper::getOutputShape(const Shape& xShape, Shape& shape) {
  cudnnTensorDescriptor_t xDesc;
  setCudnnTensor(xDesc, xShape);
  shape.resize(4);
  CUDNN_CALL(cudnnGetPooling2dForwardOutputDim(poolingDesc_,
                                               xDesc,
                                               shape.data(),
                                               shape.data() + 1,
                                               shape.data() + 2,
                                               shape.data() + 3));
  cudnnDestroyTensorDescriptor(xDesc);
}

void PoolingWrapper::forward(Tensor x, Tensor y) {
  cudaSetDevice(x->getDeviceId().no);

  cudnnTensorDescriptor_t xDesc, yDesc;
  setCudnnTensor(xDesc, x);
  setCudnnTensor(yDesc, y);

  const float alpha = 1.0f;
  const float beta = 0.0f;

  CUDNN_CALL(cudnnPoolingForward(cudnnHandle_,
                                 poolingDesc_,
                                 &alpha,
                                 xDesc,
                                 x->data(),
                                 &beta,
                                 yDesc,
                                 y->data()));
  cudnnDestroyTensorDescriptor(xDesc);
  cudnnDestroyTensorDescriptor(yDesc);
}

void PoolingWrapper::backward(Tensor x, Tensor xGrad, Tensor y, Tensor yGrad) {
  cudaSetDevice(x->getDeviceId().no);

  cudnnTensorDescriptor_t xDesc, yDesc;
  setCudnnTensor(xDesc, x);
  setCudnnTensor(yDesc, y);

  const float alpha = 1.0f;
  const float beta = 1.0f;

  CUDNN_CALL(cudnnPoolingBackward(cudnnHandle_,
                                  poolingDesc_,
                                  &alpha,
                                  yDesc,
                                  y->data(),
                                  yDesc,
                                  yGrad->data(),
                                  xDesc,
                                  x->data(),
                                  &beta,
                                  xDesc,
                                  xGrad->data()));
  cudnnDestroyTensorDescriptor(xDesc);
  cudnnDestroyTensorDescriptor(yDesc);
}

void PoolingWrapper::setPoolingDescriptor(int height,
                                          int width,
                                          int padHeight,
                                          int padWidth,
                                          int strideHeight,
                                          int strideWidth) {
  CUDNN_CALL(cudnnCreatePoolingDescriptor(&poolingDesc_));
  CUDNN_CALL(cudnnSetPooling2dDescriptor(poolingDesc_,
                                         poolingMode_,
                                         CUDNN_NOT_PROPAGATE_NAN,
                                         height,
                                         width,
                                         padHeight,
                                         padWidth,
                                         strideHeight,
                                         strideWidth));
}

PoolingWrapper::~PoolingWrapper() {
  // std::cerr << "destroy pool-wrapper" << std::endl;
  CUDNN_CALL(cudnnDestroyPoolingDescriptor(poolingDesc_));
}

#else

CUDNNWrapper::CUDNNWrapper() {
  ABORT(
      "To use convolution and pooling, recompile with CUDNN (cmake flag "
      "-DUSE_CUDNN=on)");
}

CUDNNWrapper::~CUDNNWrapper() {}

ConvolutionWrapper::ConvolutionWrapper(const Shape&,
                                       const Shape&,
                                       int,
                                       int,
                                       int,
                                       int) {
  ABORT(
      "To use convolution and pooling, recompile with CUDNN (cmake flag "
      "-DUSE_CUDNN=on)");
}

void ConvolutionWrapper::getOutputShape(const Shape&, Shape&) {
  ABORT(
      "To use convolution and pooling, recompile with CUDNN (cmake flag "
      "-DUSE_CUDNN=on)");
}

void ConvolutionWrapper::forward(Tensor, Tensor, Tensor, Tensor) {
  ABORT(
      "To use convolution and pooling, recompile with CUDNN (cmake flag "
      "-DUSE_CUDNN=on)");
}

void ConvolutionWrapper::backward(Tensor,
                                  Tensor,
                                  Tensor,
                                  Tensor,
                                  Tensor,
                                  Tensor) {
  ABORT(
      "To use convolution and pooling, recompile with CUDNN (cmake flag "
      "-DUSE_CUDNN=on)");
}

ConvolutionWrapper::~ConvolutionWrapper() {}

PoolingWrapper::PoolingWrapper(int, int, int, int, int, int, std::string) {
  ABORT(
      "To use convolution and pooling, recompile with CUDNN (cmake flag "
      "-DUSE_CUDNN=on)");
}

void PoolingWrapper::getOutputShape(const Shape&, Shape&) {
  ABORT(
      "To use convolution and pooling, recompile with CUDNN (cmake flag "
      "-DUSE_CUDNN=on)");
}

void PoolingWrapper::forward(Tensor x, Tensor y) {
  ABORT(
      "To use convolution and pooling, recompile with CUDNN (cmake flag "
      "-DUSE_CUDNN=on)");
}

void PoolingWrapper::backward(Tensor, Tensor, Tensor, Tensor) {
  ABORT(
      "To use convolution and pooling, recompile with CUDNN (cmake flag "
      "-DUSE_CUDNN=on)");
}

PoolingWrapper::~PoolingWrapper() {}

#endif
}  // namespace marian
