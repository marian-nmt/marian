#pragma once

__kernel void gCopyRows(
	__global float* out, 
	__global const float* in, 
	const unsigned int cols,
    __global const unsigned int* targetRowIdx,
    const unsigned int numPairs) 
{

}
  
