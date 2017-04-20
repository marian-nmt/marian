#pragma once
#include <memory>
#include "kernels/tensor_operators.h"
#include "kernels/cuda_helpers.h"
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <curand_kernel.h>
#include <curand.h>
#include "training/sparse_tensor.h"


namespace marian {

class GradientDropBase {
	float* feedback;
  	float* temp_d;
  	float cut_off;
  	int step;
  	int _device;

	void grad_drop_do(float* data, float* errors, float* tmp, int len, float rate){
	  int threads = 512;
	  int blocks = 1 + len/threads;
	  cudaSetDevice(_device);
	  grad_add_error<<<blocks, threads>>>(data, errors, len);
	  //full sort
	  //int sortSize = len;
	  int sortSize = min(100000, len);
	  int blocksSample = 1 + sortSize/threads;
	  randomSampling<<<blocksSample, threads>>>(data, tmp, sortSize, len / sortSize, len);
	  //dont update the cut threshold every step
	  
		thrust::device_ptr<float> dev_data_ptr(tmp);
		thrust::sort(dev_data_ptr, dev_data_ptr + sortSize ); // OVERKILL. Too slow and need extra memory. Need to replace with faster k-th selection. (inplace Radix Select?)
		int cut_index = sortSize * rate;
		if (cut_index >= sortSize)
			    cut_index = sortSize -1;
		cudaMemcpy(&cut_off, tmp + cut_index, sizeof(float), cudaMemcpyDeviceToHost);

		grad_drop<<<blocks, threads>>>(data, tmp, errors, cut_off, len);

	}

  public:

    void dropGraph(Tensor t, SparseTensor destination, double rate = 0.99) {
    	
    	cudaSetDevice(t->getDevice());
    	if(!feedback){
    		 _device = t->getDevice();
    		 cudaMalloc(&feedback, sizeof(float) * t->size());
    		 cudaMalloc(&temp_d, sizeof(float) * t->size());
    		 cudaMemset(feedback, 0, sizeof(float) * t->size());
    		 cudaMemset(temp_d, 0, sizeof(float) * t->size());
    		
    		 step = 0;
    	}
 		
    	grad_drop_do( t->data(), feedback, temp_d, t->size(), rate);
    	
    	if (rate < 0.9)
    		return;

    	thrust::device_ptr<float> mask_ptr(temp_d);
    	int denseSize = t->size();
    	thrust::inclusive_scan(mask_ptr, mask_ptr + denseSize , mask_ptr);
    	float sparseSize;
    	cudaMemcpy(&sparseSize, temp_d + denseSize - 1, sizeof(float), cudaMemcpyDeviceToHost);
    	//convert result of exscan to indices.
    	int threads = 512;
	  	int blocks = 1 + denseSize/threads;
	  	cudaSetDevice(t->getDevice());
    	buildIndices<<<blocks, threads>>>(t->data(), temp_d, destination->data(), destination->indices(),  denseSize);
    	destination->setSize(sparseSize);
    	
    	cudaStreamSynchronize(0);


    	//sanity check
    	cudaSetDevice(t->getDevice());
	    if (step < 10){
	      thrust::device_ptr<float> dev_data_ptr(t->data());
	      int x = thrust::count(dev_data_ptr, dev_data_ptr + t->size() , 0);
	      std::cerr<<"GPU :"<< t->getDevice()<<"  |  overall dropping "<<(float)x /  t->size() <<std::endl;
	    }
	    step++;
    }
};


typedef Ptr<GradientDropBase> GradientDrop;

}
