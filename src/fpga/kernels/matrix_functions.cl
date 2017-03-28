#pragma once

//////////////////////////////////////////////////////
float sumBase(__global float *input, uint count)
{
  float ret = 0.0f;
  for (uint i = 0; i < count; ++i) {
    ret += input[i];
  }
  return ret;
}

__kernel void sum(                                                    
   __global float* input, 
   __global float* output,
   const uint count)
{
  float ret = sumBase(input, count);
  (*output) = ret;
}                                      

__kernel void sum_size_t(                                                    
   __global uint* input, 
   __global uint* output,
   const uint count)
{
  uint ret = 0;
  for (uint i = 0; i < count; ++i) {
    ret += input[i];
  }
  (*output) = ret;
}                                      

//////////////////////////////////////////////////////

__kernel void gCopyRows(
	__global float* out, 
	__global const float* in, 
	const unsigned int cols,
    __global const unsigned int* targetRowIdx,
    const unsigned int numPairs) 
{
  for (unsigned int j = 0; j < numPairs; ++j) {
    unsigned int srcId = targetRowIdx[j];    
    __global float *rowOut = out + j * cols;

    __global const float *rowIn = in + srcId * cols;
   
  	for (size_t i = 0; i < cols; ++i) {
       rowOut[i] = srcId;  	
       //float f = rowIn[i];
       //rowOut[i] = f;
  	}

    
  }
  
  /*
  for (int bid = 0; bid < numPairs; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if (j < numPairs) {
      size_t dstId = j;
      size_t srcId = targetRowIdx[j];

      float* rowOut = out + dstId * cols;
      const float* rowIn = in + srcId * cols;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < cols)
          rowOut[i] = rowIn[i];
      }
    }
  }
  */
}
  
//////////////////////////////////////////////////////
  