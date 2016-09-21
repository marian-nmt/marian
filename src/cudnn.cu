#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cudnn.h>

#include <boost/timer/timer.hpp>

#include "tensor.h"
#include "tensor_operators.h"
#include "param_initializers.h"

using namespace marian;

void CudnnSoftmaxForward(cudnnHandle_t cudnnHandle,
                  Tensor out, Tensor in) {
    float alpha = 1, beta = 0;
    cudnnSoftmaxForward(cudnnHandle,
                        CUDNN_SOFTMAX_LOG,
                        CUDNN_SOFTMAX_MODE_CHANNEL,
                        &alpha,
                        in.cudnn(),
                        in.data(),
                        &beta,
                        out.cudnn(),
                        out.data());
    cudaDeviceSynchronize();
}

void CudnnSoftmaxBackward(cudnnHandle_t cudnnHandle,
                          Tensor out, Tensor in) {
    float alpha = 1, beta = 0;
    cudnnSoftmaxBackward(cudnnHandle,
                         CUDNN_SOFTMAX_LOG,
                         CUDNN_SOFTMAX_MODE_CHANNEL,
                         &alpha,
                         in.cudnn(),
                         in.data(),
                         out.cudnn(),
                         out.data(),
                         &beta,
                         out.cudnn(),
                         out.data());
    cudaDeviceSynchronize();
}

int main() {
    cudnnHandle_t cudnnHandle;
    cudnnCreate(&cudnnHandle);
    
    int d = 10;
    
    Tensor in({d, d});
    Tensor out({d, d});
    Tensor grad({d, d});
    Tensor adj({d, d}, 1);
    
    auto f = uniform(-5, 5);
    f(in);
    
    std::cerr << in.Debug() << std::endl;
    
    {
        boost::timer::cpu_timer timer;
        for(int i = 0; i < 1; ++i) {
          CudnnSoftmaxForward(cudnnHandle, out, in);
          std::cerr << out.Debug() << std::endl;
          CudnnSoftmaxBackward(cudnnHandle, grad, in);
          std::cerr << grad.Debug() << std::endl;
        }
      
        std::cerr << timer.format(5, "%ws") << std::endl;
    }
    
    {
        boost::timer::cpu_timer timer;
        for(int i = 0; i < 1; ++i) {
          Element(_1 = _2, out, in);
          Softmax(&out);
          std::cerr << out.Debug() << std::endl;
          SoftmaxGrad(grad, adj, out);
          std::cerr << grad.Debug() << std::endl; 
        }
        //std::cerr << grad.Debug() << std::endl;
        std::cerr << timer.format(5, "%ws") << std::endl;
    }
    
    
    //// Copy back
    //float *result = (float *) malloc(m * c * sizeof(float));
    //cudaMemcpy(result, d_softmaxData, m * c * sizeof(float), cudaMemcpyDeviceToHost);
    //cudaDeviceSynchronize();
    //
    //// Log
    //printf("SOFTMAX:\n");
    //printMatrix(result, c, m);
    //
    //// Try backward
    //cudnnTensorDescriptor_t diffTensorDesc;
    //cudnnCreateTensorDescriptor(&diffTensorDesc);
    //cudnnSetTensor4dDescriptor(diffTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    //                           m, c, 1, 1);
    //
    //float *d_gradData;
    //cudaMalloc((void**) &d_gradData, m * c * sizeof(float));
    //
    //float *diffData = makeDiffData(m, c);
    //float *d_diffData;
    //cudaMalloc((void**) &d_diffData, m * c * sizeof(float));
    //cudaMemcpy(d_diffData, diffData, m * c * sizeof(float), cudaMemcpyHostToDevice);
    //cudaDeviceSynchronize();
    //
    //cudnnSoftmaxBackward(cudnnHandle,
    //                     CUDNN_SOFTMAX_ACCURATE,
    //                     CUDNN_SOFTMAX_MODE_CHANNEL,
    //                     &alpha,
    //                     srcTensorDesc,
    //                     d_softmaxData,
    //                     diffTensorDesc,
    //                     d_diffData,
    //                     &beta,
    //                     sftTensorDesc,
    //                     d_gradData);
    //cudaDeviceSynchronize();
    //
    //// Copy back
    //float *result_backward = (float *) malloc(m * c * sizeof(float));
    //cudaMemcpy(result_backward, d_gradData, m * c * sizeof(float), cudaMemcpyDeviceToHost);
    //cudaDeviceSynchronize();
    //
    //// Log
    //printf("GRADIENT:\n");
    //printMatrix(result_backward, c, m);
    //
    //// Destruct
    //free(result);
    //free(diffData);
    //free(result_backward);
    //free(fcLayer);
    //
    //cudnnDestroyTensorDescriptor(srcTensorDesc);
    //cudnnDestroyTensorDescriptor(sftTensorDesc);
    //cudnnDestroyTensorDescriptor(diffTensorDesc);
    //cudaFree(d_fcLayer);
    //cudaFree(d_softmaxData);
    //cudaFree(d_gradData);
    //cudaFree(d_diffData);
    cudnnDestroy(cudnnHandle);
    return 0;
}