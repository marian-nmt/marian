#pragma once
#include <memory>
#include "kernels/tensor_operators.h"
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <curand_kernel.h>
#include <curand.h>
#define DEBUG false
#define EPS 1e-18

namespace marian {

// @TODO: use inplace Radix Select
// create actual sparse tensor class. This one is just minimal 

__global__ void gScatterUpdate(float* denseData, float* sparseData, int* sparseIndices, int denseSize, int sparseSize, int offset){
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= sparseSize)
		return;
	if (sparseIndices[idx] + offset >= 0 && sparseIndices[idx] + offset < denseSize)
		denseData[ sparseIndices[idx] + offset ] = sparseData[idx];
	else
		printf(" ------        LHOOOO offset %d --- %d %d\n", offset, sparseIndices[idx] + offset, denseSize);
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
	if (indices[idx] < 0)
		if (DEBUG) printf("\n\n!!!!!!!negative index!!!!!!!!!!!   position %d : %d , offset %d, beginning %d\n\n", idx, indices[idx], offset, indices[0] );
}

__global__ void gFindSubtensor(int* indices, int size, int targetStart, int targetEnd, int* resultStart, int* resultEnd){


	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= size)
		return;

	if (indices[idx] >= targetStart && (idx ==0 || indices[idx - 1] < targetStart)){
		resultStart[0] = idx;
	}
	if (idx > 0 && indices[idx] < indices[idx - 1]){
		if (DEBUG) printf(" ========== INVALID   ===  %d %d\n",indices[idx], indices[idx-1]);
		if (DEBUG) printf("              indexes   :  %d %d\n",idx, idx-1 );

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


	int* d_is_unsorted;
	int* gstart_;
	int *gend_;
public:
	SparseTensorBase(int capacity, size_t device){
		device_ = device;
		capacity_ = capacity;
		cudaSetDevice(device_);
		cudaMalloc(&data_, sizeof(float) * capacity);
    	cudaMalloc(&indices_, sizeof(int) * capacity);

    	cudaMalloc(&gstart_, sizeof(int) * 100);
    	cudaMalloc(&gend_, sizeof(int) * 100);



    	if (DEBUG) std::cout<<"INITIATE SPARSE TENSOR HOLDER AT GPU "<<device<<" capacity "<<capacity<<std::endl;
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

	void copyFrom(float* data, int* indices, int size, bool data_only){
		if (capacity_ < size){
			if (DEBUG) std::cerr<<"INI DIA PELAKUNYA GAN"<<std::endl;
			exit(1);
			return;
			//NO enough capacity
		}
		size_ = size;
		if(size == 0) return;
		cudaSetDevice(device_);

		cudaMemcpy(data_, data, size * sizeof(float), cudaMemcpyDefault);
		if (!data_only)
			cudaMemcpy(indices_, indices, size * sizeof(int), cudaMemcpyDefault);
		cudaStreamSynchronize(0);
	}

	void copyFrom(std::shared_ptr<SparseTensorBase> t, bool data_only = false){
		copyFrom(t->data(), t->indices(), t->size(), data_only);
	}

	void copyFromDense(Tensor t ){
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

	void toDense(Tensor t, int offset){
		cudaSetDevice(device_);
		int threads = 512;
		int blocks = 1 + size_ /threads;
		t->set(0);
		gScatterUpdate<<<blocks, threads>>> (t->data() , data_, indices_, t->size(), size_, offset);
	}

	void scatterUpdate(Tensor t, int offset){
		cudaSetDevice(device_);
		cudaStreamSynchronize(0);
		int threads = 512;
		int blocks = 1 + size_ /threads;
		gScatterUpdate<<<blocks, threads>>> (t->data() , data_, indices_, t->size(), size_ , offset);
	}

	void scatterCopyFrom(Tensor t){
		cudaSetDevice(device_);
		int threads = 512;
		int blocks = 1 + size_ /threads;
		gScatterCopy<<<blocks, threads>>> (t->data() , data_, indices_, t->size(), size_);
		cudaStreamSynchronize(0);
	}

	std::shared_ptr<SparseTensorBase> subtensor(int pos, int size, int idx){
		cudaSetDevice(device_);
		cudaStreamSynchronize(0);
		int* start = gstart_ + idx;
		int* end = gend_ + idx;


		int threads = 512;
		int blocks = 1 + size_ /threads;
		//std::cout<<"cutting dense from "<<pos<<" to "<<pos + size<<std::endl;
		cudaMemset(start, -1, sizeof(int));
		cudaMemset(end, 0, sizeof(int));

		gFindSubtensor<<<blocks, threads>>> (indices_ , size_, pos, pos + size, start, end);
		
		int startOffset;
		int endOffset;
		int tmp_dt;
		cudaMemcpy(&startOffset, start, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(&endOffset, end, sizeof(int), cudaMemcpyDeviceToHost);
		
		if (startOffset != -1 && startOffset < size_) cudaMemcpy(&tmp_dt, indices_ + startOffset , sizeof(int), cudaMemcpyDeviceToHost);


		if (startOffset == -1 || startOffset >= size_ || tmp_dt < pos){
			if (DEBUG) std::cout<<"NO SUBTENSOR FOUND"<<std::endl;
			if (DEBUG) std::cout<<"kena di "<<startOffset << " looking for "<<pos<<std::endl;
			if (startOffset != -1 && startOffset < size_) {
				int prev;
				int next;

				cudaMemcpy(&prev, indices_ + startOffset -1 , sizeof(int), cudaMemcpyDeviceToHost);
				cudaMemcpy(&next, indices_ + startOffset +1, sizeof(int), cudaMemcpyDeviceToHost);
				if (DEBUG) printf("element2 yang mengapit posisi: %d %d %d\n",prev, tmp_dt, next);

			}
			startOffset = endOffset + 1;
		}
		if (startOffset > endOffset){
			if (DEBUG) std::cout<<" BLEBER BLEBER BLEBER BLEBER BLEBER BLEBER BLEBER BLEBER BLEBER BLEBER BLEBER BLEBER BLEBER BLEBER "<<std::endl;
			if (DEBUG) std::cout<<startOffset<<"   "<<endOffset<< std::endl;

		}

		//std::cout<<"got cut: "<<startOffset<<" to "<<endOffset<<std::endl;

		int subtensorSize = endOffset - startOffset + 1;
		cudaStreamSynchronize(0);
		return std::shared_ptr<SparseTensorBase>( new SparseTensorBase(data_ + startOffset, indices_ + startOffset, subtensorSize, device_) );
	}
};

typedef std::shared_ptr<SparseTensorBase> SparseTensor;



__global__ void grad_drop(float* data, float* tmp, float* errors, float cut_off, int max_size){
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= max_size)
    return;
  if (std::abs(data[idx])  <= cut_off){ // get the scaling
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
  if (idx >= denseSize)
    return;
  
  int t_id = (int) (denseSum[idx] + 0.2) -1;
  if (t_id < 0){
  	return;
  }
  
  if (idx == 0 && denseSum[idx] > 0 + 0.5){
  	sparseIndices[ t_id ] = idx;
  	sparseData[ t_id ] = denseData[idx];
  }
  else if (idx > 0 && denseSum[idx] - denseSum[idx-1] > 0.5){
  	sparseIndices[ t_id ] = idx;
  	sparseData[ t_id ] = denseData[idx];
  }
}

__global__ void randomSampling(float* originalData, float* data, int size, int scale, int fullSize){
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
 	if (idx >= size)
    	return;
    //printf("data %d full-size %d, scale %d\n", idx * scale, fullSize, scale);
    //data[idx] = abs(originalData[ idx * scale ] / (EPS + params[idx * scale]) ); //scaling
    data[idx] = abs(originalData[ idx * scale ] ); //no scaling

}



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
	  int sortSize = len;
	  //int sortSize = min(5000, len);
	  //cudaMemcpy(tmp, data, len * sizeof(float), cudaMemcpyDeviceToDevice);
	  int blocksSample = 1 + sortSize/threads;
	  
	  randomSampling<<<blocksSample, threads>>>(data, tmp, sortSize, len / sortSize, len);
	  //full_abs<<<blocksSample, threads>>>(tmp,sortSize);

	  //dont update the cut threshold every step
	  
		cudaSetDevice(_device);
		thrust::device_ptr<float> dev_data_ptr(tmp);
		thrust::sort(dev_data_ptr, dev_data_ptr + sortSize ); // OVERKILL. Too slow and need extra memory. Need to replace with faster k-th selection. (inplace Radix Select?)
		int cut_index = sortSize * rate;
		if (cut_index >= sortSize)
			    cut_index = sortSize -1;
		cudaMemcpy(&cut_off, tmp + cut_index, sizeof(float), cudaMemcpyDeviceToHost);
		//std::cout<<cut_off<<std::endl;  
		  

		grad_drop<<<blocks, threads>>>(data, tmp, errors, cut_off, len);

	}

  public:

    int wow;
    void dropGraph(Tensor t, SparseTensor destination, double rate = 0.99) {

    	
    	cudaSetDevice(t->getDevice());
    	if(!feedback){
    		 _device = t->getDevice();
    		 cudaMalloc(&feedback, sizeof(float) * t->size());
    		 cudaMalloc(&temp_d, sizeof(float) * t->size());
    		 wow = 0;
    		 step = 0;
    		 std::cerr<<"MALLOC for grad Dropper GPU "<<t->getDevice()<<std::endl;
    	}
 		/*
    	if (partial)
    	for(auto& param : graph->params()){
    		int len = param->grad()->size();
    		grad_drop_do(param->grad()->data(), param->val()->data(),  feedback + offset, temp_d + offset, len, rate);
    		offset += len;
    	}
    	if (!partial){*/
    	grad_drop_do( t->data(), feedback, temp_d, t->size(), rate);
    	//}

    	//TODO: convert params().grads()->data() to sparse tensor in sparseDestination:
    	// exclusive scan, copy index
    	
    	if (rate < 0.9)
    		return;

    	thrust::device_ptr<float> mask_ptr(temp_d);
    	int denseSize = t->size();
    	thrust::inclusive_scan(mask_ptr, mask_ptr + denseSize , mask_ptr);
    	float sparseSize;
    	cudaMemcpy(&sparseSize, temp_d + denseSize - 1, sizeof(float), cudaMemcpyDeviceToHost);
    	//std::cout<<"GRAD DROP, gradient in gpu: "<<graph->params().grads()->getDevice()<<" sparse in gpu "<< destination->getDevice()<<std::endl;
    	//convert result of exscan to indices.
    	int threads = 512;
	  	int blocks = 1 + denseSize/threads;
	  	cudaSetDevice(t->getDevice());
    	buildIndices<<<blocks, threads>>>(t->data(), temp_d, destination->data(), destination->indices(),  denseSize);
    	destination->setSize(sparseSize);
    	
    	/*
    	cudaMemcpy(cpu_tmp, destination->indices(), sizeof(int) * (int)sparseSize , cudaMemcpyDeviceToHost);
    	cudaMemcpy(f_cpu_tmp, temp_d, sizeof(float) * 300 , cudaMemcpyDeviceToHost);

    	for (int i=1;i<(int) sparseSize;i++){
    		if (cpu_tmp[i] < cpu_tmp[i-1])
    			printf("INVALID %d %d (index %d %d)\n", cpu_tmp[i], cpu_tmp[i-1], i , i-1);
    	}
    	//sample
    	for (int i=0;i<25;i++)
    		printf(" %d ", cpu_tmp[i]);
    	printf("\n");*/
    	
    	cudaStreamSynchronize(0);


    	//sanity check
    	cudaSetDevice(t->getDevice());
	    if (wow < 10 || wow % 1000 == 0){
	      thrust::device_ptr<float> dev_data_ptr(t->data());
	      int x = thrust::count(dev_data_ptr, dev_data_ptr + t->size() , 0);
	      std::cerr<<"GPU :"<< t->getDevice()<<"  |  overall dropping "<<(float)x /  t->size() <<std::endl;
	      //std::cerr<<(int)sparseSize <<"/" <<denseSize<<std::endl;
	    }
	    step++;
	    wow++;
    }

    void dropGraph(Ptr<ExpressionGraph> graph, SparseTensor destination, double rate = 0.99){
    	dropGraph(graph->params().grads(), destination, rate );
    }
};


typedef Ptr<GradientDropBase> GradientDrop;

}
