#pragma once
#include "kernels/tensor_operators.h"
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

namespace marian {

// @TODO: modify computation graph to group all paramters in single matrix object.
// This will allow to perform a single large SGD update per batch. Currently there
// are as many updates as different parameters.

__global__ void grad_drop(float* data, float* errors, float cut_off, int max_size){
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= max_size)
    return;
  if (std::abs(data[idx])  <= cut_off){
    errors[idx] = data[idx];
    data[idx] = 0;
  }else{
    errors[idx] = 0;
  }
}

__global__ void grad_add_error(float* data, float* errors, int max_size){
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= max_size)
    return;
  data[idx] += errors[idx];
}

__global__ void full_abs(float* data, int max_size){
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= max_size)
    return;
  data[idx] = abs(data[idx]);
}


class GradientDrop {
	float* feedback;
  	float* temp_d;

	void grad_drop_do(float* data, float* errors, int len, float rate){
	  int threads = 512;
	  int blocks = 1 + len/threads;
	  grad_add_error<<<blocks, threads>>>(data, errors, len);

	  cudaMemcpy(temp_d, data, len * sizeof(float), cudaMemcpyDeviceToDevice);
	  full_abs<<<blocks, threads>>>(temp_d,len);

	  thrust::device_ptr<float> dev_data_ptr(temp_d);
	  thrust::sort(dev_data_ptr, dev_data_ptr + len); // OVERKILL. Too slow and need extra memory. Need to replace with faster k-th selection. (inplace Radix Select?)
	  int cut_index = len * rate;
	  if (cut_index >= len)
	    cut_index = len -1;
	  float cut_off;
	  cudaMemcpy(&cut_off, temp_d + cut_index, sizeof(float), cudaMemcpyDeviceToHost);
	  
	  grad_drop<<<blocks, threads>>>(data, errors, cut_off, len);

	  /*
	  if (wow % 2000 == 0 || wow < 10){
	    cudaMemcpy(gray, data, len * sizeof(float), cudaMemcpyDeviceToHost);
	  
	    int  x = 0;
	    for (int i=0;i< len;i++){
	      if (gray[i] == 0 )
	        x++;
	    }

	    std::cerr<<"dropping "<<(float)x / len<<std::endl;
	  }*/
	}

  public:
    void dropTensor(Tensor t, double rate) {
    	if(!feedback){
    		 cudaMalloc((void**)&feedback, sizeof(float) * t->size());
    		 cudaMalloc((void**)&temp_d, sizeof(float) * t->size());
    	}
    	grad_drop_do(t->data(), feedback, t->size() , rate);
    }
};


}
