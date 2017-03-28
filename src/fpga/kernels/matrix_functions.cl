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
	const uint cols,
  __global const uint* targetRowIdx,
  const uint numPairs) 
{
  for (uint j = 0; j < numPairs; ++j) {
    uint srcId = targetRowIdx[j];    
    __global float *rowOut = out + j * cols;

    __global const float *rowIn = in + srcId * cols;
   
  	for (uint i = 0; i < cols; ++i) {
       //rowOut[i] = srcId;  	
       //float f = rowIn[i];
       rowOut[i] = srcId;
  	}

    
  }
  
}
  
//////////////////////////////////////////////////////
  