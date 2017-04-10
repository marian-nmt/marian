#pragma once

/////////////////////////////////////////////////////////////////////////////

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

/////////////////////////////////////////////////////////////////////////////

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

    uint inOffset =  srcId * cols;
    __global const float *rowIn = in + inOffset;
    
  	for (uint i = 0; i < cols; ++i) {
       //rowOut[i] = srcId;  	
       float f = rowIn[i];
       rowOut[i] = f;
  	}

    //const float f = cols;
    //rowOut[0] = f;
    
  }
  
}
  
/////////////////////////////////////////////////////////////////////////////
  
__kernel void transpose(
  __global float* out, 
  __global const float* in, 
  const uint rows,
  const uint cols)
{
  uint i = 0;
  for (uint row = 0; row < rows; ++row) {
    for (uint col = 0; col < cols; ++col) {
      float v = in[i];
      
      //uint outInd = row * cols + col;
      uint outInd = col * rows + row;
      out[outInd] = v;
      
      ++i;
    }
  }
}

/////////////////////////////////////////////////////////////////////////////

__kernel void prod(
  __global float* C, 
  __global const float* A, 
  __global const float* B, 
  const uint rowsA,
  const uint colsA,
  const uint rowsB,
  const uint colsB)
{
  for (uint rowA = 0; rowA < rowsA; ++rowA) {
    for (uint colB = 0; colB < colsB; ++colB) {
      float sum = 0;
      
      for (uint colA = 0; colA < colsA; ++colA) {
        float valA = A[rowA * colsA + colA];
        float valB = B[colA * colsB + colB];
        sum += valA * valB;
      }
      
      C[rowA * colsB + colB] = sum; 
    }
  }
  
}

/////////////////////////////////////////////////////////////////////////////

__kernel void gElementwiseOps(__global float* out,
                                __global const float* state,
                                __global const float* ruh,
                                __global const float* t,
                                __global const float* b,
                                __global const float* bx1,
                                __global const float* bx2,
                                uint rows, uint cols) 
{
}

/////////////////////////////////////////////////////////////////////////////

__kernel void gElementTanh(__global float* out, 
                          __global const float* in1, 
                          __global const float* in2,
                          uint rows, uint cols) 
{
  uint noElements = rows * cols;
  for (uint i = 0; i < noElements; ++i) {
    out[i] = tanh(out[i] + in1[i] * in2[i]);
  }
}

/////////////////////////////////////////////////////////////////////////////

__kernel void gElementWhatever(__global float* out, 
                          __global const float* in1, 
                          __global const float* in2,
                          uint rows, uint cols) 
{
  uint noElements = rows * cols;
  for (uint i = 0; i < noElements; ++i) {
    // (1.0 - _1) * _2 + _1 * _3
    out[i] = (1.0f - out[i]) * in1[i] + out[i] * in2[i];
  }
}

/////////////////////////////////////////////////////////////////////////////

__kernel void gBroadcastVecAdd(__global float* out, 
                              __global const float* in, 
                              uint rows, uint cols) 
{
  for (uint noColumn = 0; noColumn < cols; ++noColumn) {
    float vecValue = in[noColumn];
  
    uint index = noColumn;
    for (uint noRow = 0; noRow < rows; ++noRow) {
        out[index] += vecValue;
        index += cols;
    }
  
  }

}

/////////////////////////////////////////////////////////////////////////////

__kernel void gBroadcastVecTanh(__global float* out, 
                              __global const float* in, 
                              uint rows, uint cols) 
{
  for (uint noColumn = 0; noColumn < cols; ++noColumn) {
    float vecValue = in[noColumn];
  
    uint index = noColumn;
    for (uint noRow = 0; noRow < rows; ++noRow) {
        out[index] = tanh(out[index] + vecValue);
        index += cols;
    }
  
  }

}

/////////////////////////////////////////////////////////////////////////////

__kernel void gBroadcastTanh(__global float* out, 
                            __global const float* in1, 
                            __global const float* in2,
                            uint srcSize, 
                            uint sumBeams, 
                            uint cols, 
                            __global const int* batchMapping,
                           uint batchMappingSize, 
                           uint outSize, 
                           uint in1Size, 
                           uint in2Size,
                           uint inRows)
{
  uint maxId = srcSize * inRows * cols;
  for (uint id = 0; id < maxId; ++id) {
    int row = id / cols;
    int stateIdx = id % cols;

    int beamIdx = row / srcSize;
    int srcId = row % srcSize;

    int batchIdx = batchMapping[beamIdx];
  
    //assert(id < outSize);
    //assert((batchIdx * srcSize + srcId) * cols + stateIdx < in1Size);
    //assert(beamIdx * cols + stateIdx < in2Size);
  
    float x = in1[(batchIdx * srcSize + srcId) * cols + stateIdx];
    float y = in2[beamIdx * cols + stateIdx];
    out[id] = tanh(x + y);
  }
  
}

/////////////////////////////////////////////////////////////////////////////

__kernel void gLogit(__global float* out, 
                     __global const float* in, 
                     uint rows, uint cols) 
{
  uint i = 0;
  
  for (uint noColumn = 0; noColumn < cols; ++noColumn) {
    for (uint noRow = 0; noRow < rows; ++noRow) {
      float p = out[i] + in[i];
      out[i] = 1.0f / (1.0f + exp(-p));
      ++i;
    }
  }  
}  

/////////////////////////////////////////////////////////////////////////////

__kernel void gSlice(__global float* out, 
                    __global const float* in,
                   uint n, uint dim,
                   uint rows, uint cols) 
{
  uint offset = n * dim;

  for (uint row = 0; row < rows; ++row) {
    for (uint outCol = 0; outCol < dim; ++outCol) {
      uint inCol = offset + outCol;
      
      uint inInd = row * cols + inCol;
      uint outInd = row * dim + outCol;
      
      out[outInd] = in[inInd];
    }
  }
}

/////////////////////////////////////////////////////////////////////////////

__kernel void gPasteRows(__global float* d_out, 
                    uint outRows, 
                    uint outCols, 
                    __global const float* d_in, 
                    uint inRows, 
                    uint inCols, 
                    uint colNo, 
                    uint sparse) 
{
  uint maxId = inRows * inCols;
  for (uint id = 0; id < maxId; ++id) {
    uint inRow = id / inCols;
    uint inCol = id % inCols;
    uint outID = (outRows + sparse * inRow) * outCols + inCol + colNo;
    d_out[outID] = d_in[id];
  }
}

/////////////////////////////////////////////////////////////////////////////

__kernel void gMapMatrix(__global float* d_in, 
                    uint numRows, 
                    uint numCols, 
                    uint mappingCols, 
                    __global const int* mapping, 
                    uint i) 
{

}

/////////////////////////////////////////////////////////////////////////////

__kernel void gMean(__global float* d_out, 
                    __global const float* d_in, 
                    __global const int* mapping,
                    uint batchNum, uint senLen, uint stateLength) 
{
  for (uint id = 0; id < stateLength; ++id) {
    float sum = 0.0f;
    int counter = 0;

      
    for (int i = 0; i < batchNum * senLen; ++i) {
      sum += mapping[i] * d_in[i * stateLength + id];
      counter += mapping[i];

      if ((i + 1) % senLen == 0) {
        sum /= counter;
        d_out[(i / senLen) * stateLength + id] = sum;
        sum = 0.0f;
        counter = 0;
      }
    }
  }
}


          