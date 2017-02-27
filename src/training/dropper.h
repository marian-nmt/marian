#pragma once
#include <memory>
#include "kernels/tensor_operators.h"
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

namespace marian {

// @TODO: use inplace Radix Select
// create actual sparse tensor class. This one is just minimal 

__global__ void gScatterUpdate(float* denseData, float* sparseData, int* sparseIndices, int denseSize, int sparseSize){
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= sparseSize)
		return;
	if (sparseIndices[idx]  >= 0 && sparseIndices[idx] < denseSize)
		denseData[ sparseIndices[idx] ] = sparseData[idx];
}

__global__ void gScatterCopy(float* denseData, float* sparseData, int* sparseIndices, int denseSize, int sparseSize){
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= sparseSize)
		return;
	if (sparseIndices[idx]  >= 0 && sparseIndices[idx] < denseSize)
		sparseData[idx] = denseData[ sparseIndices[idx] ];
}

__global__ void gShift(int* indices, int size, int offset){
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= size)
		return;
	indices[idx] += offset;	
}

__global__ void gFindSubtensor(int* indices, int size, int targetStart, int targetEnd, int* resultStart, int* resultEnd){
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= size)
		return;

	if (indices[idx] >= targetStart && (idx ==0 || indices[idx - 1] < targetStart)){
		resultStart[0] = idx;
	}
	if (indices[idx] < targetEnd && (idx == size - 1 || indices[idx + 1] >= targetEnd))
		resultEnd[0] = idx;
}


class SparseTensorBase: public std::enable_shared_from_this<SparseTensorBase> {
	float *data_;
	int *indices_;
	int size_;
	int capacity_;
	size_t device_; 


	int* start;
	int* end;

public:
	SparseTensorBase(int capacity, size_t device){
		device_ = device;
		capacity_ = capacity;
		cudaSetDevice(device_);
		cudaMalloc((void**)&data_, sizeof(float) * capacity);
    	cudaMalloc((void**)&indices_, sizeof(int) * capacity);

		cudaMalloc((void**)&start, sizeof(int));
		cudaMalloc((void**)&end, sizeof(int));

    	std::cout<<"INITIATE SPARSE TENSOR HOLDER AT GPU "<<device<<" capacity "<<capacity<<std::endl;
	}

	SparseTensorBase(float* data, int* indices, int size, size_t device){
		data_ = data;
		indices_ = indices;
		size_ = size;
		capacity_ = size;
		device_ = device;
	}

	~SparseTensorBase(){

	}

	int capacity(){
		return capacity_;
	}

	int size(){
		return size_;
	}

	float* data(){
		return data_;
	}

	int* indices(){
		return indices_;
	}

	void copyFrom(float* data, int* indices, int size){
		if (capacity_ < size){
			std::cerr<<"INI DIA PELAKUNYA GAN"<<std::endl;
			exit(1);
			return;
			//NO enough capacity
		}
		size_ = size;

		cudaSetDevice(device_);

		cudaMemcpy(data_, data, size * sizeof(float), cudaMemcpyDefault);
		cudaMemcpy(indices_, indices, size * sizeof(int), cudaMemcpyDefault);
		cudaStreamSynchronize(0);
	}

	void copyFrom(std::shared_ptr<SparseTensorBase> t){
		copyFrom(t->data(), t->indices(), t->size());
	}

	void copyFromDense(Tensor t){
		cudaSetDevice(device_);	

	}

	size_t getDevice(){
		return device_;
	}

	void setSize(int size){
		size_ = size;
	}

	void shiftIndices(int offset){
		cudaSetDevice(device_);
		int threads = 512;
		int blocks = 1 + size_ /threads;
		gShift<<<blocks, threads>>> (indices_, size_, offset);
	}

	void toDense(Tensor t){
		cudaSetDevice(device_);
		int threads = 512;
		int blocks = 1 + size_ /threads;
		t->set(0);
		gScatterUpdate<<<blocks, threads>>> (t->data() , data_, indices_, t->size(), size_);
	}

	void scatterUpdate(Tensor t){
		cudaSetDevice(device_);
		int threads = 512;
		int blocks = 1 + size_ /threads;
		gScatterUpdate<<<blocks, threads>>> (t->data() , data_, indices_, t->size(), size_);
	}

	void scatterCopyFrom(Tensor t){
		cudaSetDevice(device_);
		int threads = 512;
		int blocks = 1 + size_ /threads;
		gScatterCopy<<<blocks, threads>>> (t->data() , data_, indices_, t->size(), size_);
	}

	std::shared_ptr<SparseTensorBase> subtensor(int pos, int size){
		cudaSetDevice(device_);

		int threads = 512;
		int blocks = 1 + size_ /threads;
		//std::cout<<"cutting dense from "<<pos<<" to "<<pos + size<<std::endl;
		gFindSubtensor<<<blocks, threads>>> (indices_ , size_, pos, pos + size, start, end);
		
		int startOffset;
		int endOffset;
		cudaMemcpy(&startOffset, start, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(&endOffset, end, sizeof(int), cudaMemcpyDeviceToHost);

		//std::cout<<"got cut: "<<startOffset<<" to "<<endOffset<<std::endl;

		int subtensorSize = endOffset - startOffset + 1;

		return std::shared_ptr<SparseTensorBase>( new SparseTensorBase(data_ + startOffset, indices_ + startOffset, subtensorSize, device_) );
	}
};

typedef std::shared_ptr<SparseTensorBase> SparseTensor;



__global__ void grad_drop(float* data, float* tmp, float* errors, float cut_off, int max_size){
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= max_size)
    return;
  if (std::abs(data[idx])  <= cut_off){
    errors[idx] = data[idx];
    data[idx] = 0;
    tmp[idx] = 0;
  }else{
    errors[idx] = 0;
    tmp[idx] = 1;
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

__global__ void buildIndices(float* denseData, float* denseSum, float* sparseData, int* sparseIndices, int denseSize){
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int t_id = (int) (denseSum[idx] + 0.2) -1;
  if (idx >= denseSize || t_id < 0)
    return;
  if (idx == 0 && denseSum[idx] > 0){
  	sparseIndices[ t_id ] = idx;
  	sparseData[ t_id ] = denseData[idx];
  }
  else if (denseSum[idx] - denseSum[idx-1] > 0.5){
  	sparseIndices[ t_id ] = idx;
  	sparseData[ t_id ] = denseData[idx];
  }
}


class GradientDrop {
	float* feedback;
  	float* temp_d;
  	float cut_off;
  	int step;

	void grad_drop_do(float* data, float* errors, float* tmp, int len, float rate){
	  int threads = 512;
	  int blocks = 1 + len/threads;
	  grad_add_error<<<blocks, threads>>>(data, errors, len);

	  cudaMemcpy(tmp, data, len * sizeof(float), cudaMemcpyDeviceToDevice);
	  full_abs<<<blocks, threads>>>(tmp,len);

	  //dont update the cut threshold every step
	  if (step < 1000 || step % 1 == 0){
		  thrust::device_ptr<float> dev_data_ptr(tmp);
		  thrust::sort(dev_data_ptr, dev_data_ptr + len); // OVERKILL. Too slow and need extra memory. Need to replace with faster k-th selection. (inplace Radix Select?)
		  int cut_index = len * rate;
		  if (cut_index >= len)
		    cut_index = len -1;
		  cudaMemcpy(&cut_off, tmp + cut_index, sizeof(float), cudaMemcpyDeviceToHost);
	  }
	  
	  grad_drop<<<blocks, threads>>>(data, tmp, errors, cut_off, len);

	}

  public:
    void dropTensor(Tensor t, float* feedback_loc, double rate = 0.99) {
    	//grad_drop_do(t->data(), feedback_loc , t->size() , rate);
    }
    int wow;
    void dropGraph(Ptr<ExpressionGraph> graph, SparseTensor destination, double rate = 0.99) {

    	
    	cudaSetDevice(graph->params().grads()->getDevice());
    	if(!feedback){
    		 cudaMalloc((void**)&feedback, sizeof(float) * graph->params().vals()->size());
    		 cudaMalloc((void**)&temp_d, sizeof(float) * graph->params().vals()->size());
    		 wow = 0;
    		 step = 0;
    		 std::cerr<<"MALLOC for grad Dropper GPU "<<graph->params().grads()->getDevice()<<std::endl;
    	}
    	int offset = 0;
    	for(auto& param : graph->params()){
    		int len = param->grad()->size();
    		grad_drop_do(param->grad()->data(), feedback + offset, temp_d + offset, len, rate);
    		offset += len;
    	}

    	//TODO: convert params().grads()->data() to sparse tensor in sparseDestination:
    	// exclusive scan, copy index
    	thrust::device_ptr<float> mask_ptr(temp_d);
    	int denseSize = graph->params().vals()->size();
    	thrust::inclusive_scan(mask_ptr, mask_ptr + denseSize , mask_ptr);
    	float sparseSize;
    	cudaMemcpy(&sparseSize, temp_d + denseSize - 1, sizeof(float), cudaMemcpyDeviceToHost);
    	//std::cout<<"GRAD DROP, gradient in gpu: "<<graph->params().grads()->getDevice()<<" sparse in gpu "<< destination->getDevice()<<std::endl;
    	//convert result of exscan to indices.
    	int threads = 512;
	  	int blocks = 1 + denseSize/threads;
	  	cudaSetDevice(graph->params().grads()->getDevice());
    	buildIndices<<<blocks, threads>>>(graph->params().grads()->data(), temp_d,destination->data(), destination->indices(),  denseSize);
    	destination->setSize(sparseSize);
    	
    	cudaStreamSynchronize(0);


    	//sanity check
	    if (wow < 10 || wow % 1000 == 0){
	      thrust::device_ptr<float> dev_data_ptr(graph->params().grads()->data());
	      int x = thrust::count(dev_data_ptr, dev_data_ptr + graph->params().vals()->size() , 0);
	      std::cerr<<"overall dropping "<<(float)x /  graph->params().vals()->size() <<std::endl;
	      std::cerr<<(int)sparseSize <<"/" <<denseSize<<std::endl;
	    }
	    step++;
	    wow++;
    }
};


}
