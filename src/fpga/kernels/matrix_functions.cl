#pragma once

__kernel void gCopyRows(
	float* out, 
	const float* in, 
	size_t cols,
    const size_t* targetRowIdx, 
    size_t numPairs) 
{
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
  
