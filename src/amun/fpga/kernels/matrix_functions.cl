
/////////////////////////////////////////////////////////////////////////////

float sumBase(__global float *input, uint count)
{
  float ret = 0.0f;
  for (uint i = 0; i < count; ++i) {
    ret += input[i];
  }
  return ret;
}

__kernel void sum_float(                                                    
   __global float* restrict input, 
   __global float* restrict output,
   const uint count)
{
  float ret = sumBase(input, count);
  (*output) = ret;
}                                      

__kernel void sum_uint(                                                    
   __global uint* restrict input, 
   __global uint* restrict output,
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
	__global float* restrict out, 
	__global const float* restrict in, 
	const uint cols,
  __global const uint* restrict targetRowIdx,
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
  __global float* restrict out, 
  __global const float* restrict in, 
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
  __global float* restrict C, 
  __global const float* restrict A, 
  __global const float* restrict B, 
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

__kernel void gElementwiseOps(__global float* restrict out,
                                __global const float* restrict state,
                                __global const float* restrict ruh,
                                __global const float* restrict t,
                                __global const float* restrict b,
                                __global const float* restrict bx1,
                                __global const float* restrict bx2,
                                uint rows, uint cols) 
{
}

/////////////////////////////////////////////////////////////////////////////

__kernel void gElementTanh(__global float* restrict out, 
                          __global const float* restrict in1, 
                          __global const float* restrict in2,
                          uint rows, uint cols) 
{
  uint noElements = rows * cols;
  for (uint i = 0; i < noElements; ++i) {
    out[i] = tanh(out[i] + in1[i] * in2[i]);
  }
}

/////////////////////////////////////////////////////////////////////////////

__kernel void gElementTanh2(__global float* restrict out, 
                          __global const float* restrict in1, 
                          __global const float* restrict in2,
                          uint rows, uint cols) 
{
  uint noElements = rows * cols;
  for (uint i = 0; i < noElements; ++i) {
    out[i] = tanh(out[i] + in1[i] + in2[i]);
  }
}

/////////////////////////////////////////////////////////////////////////////

__kernel void gElementWhatever(__global float* restrict out, 
                          __global const float* restrict in1, 
                          __global const float* restrict in2,
                          uint rows, uint cols) 
{
  uint noElements = rows * cols;
  for (uint i = 0; i < noElements; ++i) {
    // (1.0 - _1) * _2 + _1 * _3
    out[i] = (1.0f - out[i]) * in1[i] + out[i] * in2[i];
  }
}

/////////////////////////////////////////////////////////////////////////////

__kernel void gElementAddWeighted(__global float* restrict out, 
                          __global const float* restrict in, 
                          uint rows, uint cols, float weight) 
{
  uint noElements = rows * cols;
  for (uint i = 0; i < noElements; ++i) {
    // _1 + weights_.at(scorers[i]->GetName()) * _2
    out[i] = out[i] + weight * in[i];
  }
}

/////////////////////////////////////////////////////////////////////////////

__kernel void gBroadcastVecAdd(__global float* restrict out, 
                              __global const float* restrict in, 
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

__kernel void gBroadcastVecTanh(__global float* restrict out, 
                              __global const float* restrict in, 
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

__kernel void gBroadcastTanh(__global float* restrict out, 
                            __global const float* restrict in1, 
                            __global const float* restrict in2,
                            uint srcSize, 
                            uint cols, 
                            __global const int* restrict batchMapping,
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

__kernel void gBroadcastVecColumnAddWeighted(
                     __global float* restrict out, 
                     __global const float* restrict in, 
                     uint rows, uint cols,
                     float weight) 
{
  for (uint noColumn = 0; noColumn < cols; ++noColumn) {
      int index = noColumn;
      for (int noRow = 0; noRow < rows; ++noRow) {
        out[index] = weight * out[index] + in[noRow];
        index += cols;
    }

  }
  
}


/////////////////////////////////////////////////////////////////////////////

__kernel void gLogit(__global float* restrict out, 
                     __global const float* restrict in, 
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

__kernel void gSlice(__global float* restrict out, 
                    __global const float* restrict in,
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

__kernel void gPasteRows(__global float* restrict d_out, 
                    uint outRows, 
                    uint outCols, 
                    __global const float* restrict d_in, 
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

__kernel void gMapMatrix(__global float* restrict d_in, 
                    uint numRows, 
                    uint numCols, 
                    uint mappingCols, 
                    __global const int* restrict mapping, 
                    uint i) 
{
  for (uint batchIdx = 0; batchIdx < numRows; ++batchIdx) {
    int isWord = mapping[mappingCols * batchIdx + i];
    if (isWord == 0) {
      // blank out word
      for (uint j = 0; j < numCols; ++j) {
        uint ind = batchIdx * numCols + j;
        d_in[ind] = 0;
      }
    }
  }
}

/////////////////////////////////////////////////////////////////////////////

__kernel void gMean(__global float* restrict d_out, 
                    __global const float* restrict d_in, 
                    __global const int* restrict mapping,
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

/////////////////////////////////////////////////////////////////////////////

__kernel void gSoftMax(__global float* restrict softMaxP, 
                       uint rows, 
                       uint cols,
                       __global const int* restrict batchIds,
                       uint batchNum,
                       __global const int* restrict srcMapping,
                       uint srcNum) 
{
  for (uint row = 0; row < rows; ++row) {
    uint indRow = row * cols;
    
    int batchId = batchIds[row];
    uint startBatchInd = batchId * srcNum;
     
    // EXP
    float sumExp = 0;
    for (uint col = 0; col < cols; ++col) {
      int wordExist = srcMapping[startBatchInd + col];
      
      if (wordExist) {
        float val = softMaxP[indRow + col];
        val = exp(val);
        
        sumExp += val;
        softMaxP[indRow + col] = val;
      }
      else {
        softMaxP[indRow + col] = 0;
      }
    }
    
    // NORMALIZE
    for (uint col = 0; col < cols; ++col) {
      softMaxP[indRow + col] /= sumExp;
    }    
  }
}                         

/////////////////////////////////////////////////////////////////////////////

__kernel void gLogSoftMax(__global float* restrict softMaxP, 
                       uint rows, 
                       uint cols)
{
  // probably only work for non-batch
  for (uint row = 0; row < rows; ++row) {
    uint indRow = row * cols;

    // EXP
    float sumExp = 0;
    for (uint col = 0; col < cols; ++col) {
      float val = softMaxP[indRow + col];
      val = exp(val);
      
      sumExp += val;
      softMaxP[indRow + col] = val;
    }
    
    // NORMALIZE
    for (uint col = 0; col < cols; ++col) {
      float val = softMaxP[indRow + col];
      val /= sumExp;
      
      softMaxP[indRow + col] = log(val);
    }    
  }
  
}

/////////////////////////////////////////////////////////////////////////////

__kernel void gWeightedMean(__global float* restrict d_out, 
                            __global const float* restrict weights, 
                            __global const float* restrict d_in, 
                            __global const int* restrict mapping,
                            uint numRows, uint numCols, uint srcLen) 
{

  uint size = numRows * numCols;
  for (uint id = 0; id < size; ++id) {
    uint rowNo = id / numCols;
    uint batchNo = mapping[rowNo];
    uint statePos = id % numCols;
  
    float sum = 0.0f;
    for (uint i = 0; i < srcLen; ++i) {
      sum += weights[rowNo * srcLen + i] * d_in[batchNo * srcLen * numCols + (i * numCols) + statePos];
    }

    d_out[id] = sum;
  
  }

}

/////////////////////////////////////////////////////////////////////////////

__kernel void gMaxElement(
								__global float* restrict d_out, 
								__global int* restrict d_ind, 
								__global float* restrict d_in, 
								int numBatches, 
								__global int* restrict batchFirstElementIdxs)
{

}

/////////////////////////////////////////////////////////////////////////////

void insertValue(
                __global float* restrict bestCost,
                __global unsigned* restrict bestInd,
                uint count,
                float val,
                uint insertInd,
                uint probStart)
{
  uint ind = count;
  for (uint i = 1; i < count; ++i) {
    if (val <= bestCost[i]) {
      ind = i;
      break;
    }
  }
  
  // shift everything up 1
  for (uint i = ind; i < count; ++i) {
    bestCost[i+1] = bestCost[i];
    bestInd[i+1] = bestInd[i];
  }
  
  // insert value into place
  bestCost[ind] = val;
  bestInd[ind] = insertInd + probStart;
}

void replaceValueOrDiscard(
                __global float* restrict bestCost,
                __global unsigned* restrict bestInd,
                uint count,
                float val,
                uint insertInd,
                uint probStart)
{
  if (val < bestCost[0]) {
    // too low
    return;
  }
  
  uint ind = count - 1;
  for (uint i = 1; i < count; ++i) {
    if (val <= bestCost[i]) {
      ind = i - 1;
      break;
    }
  }
  
  // shift lowest value out of the array
  for (uint i = 0; i < ind; ++i) {
    bestCost[i] = bestCost[i+1];
    bestInd[i] = bestInd[i+1];
  }
  
  // insert value into place
  bestCost[ind] = val;
  bestInd[ind] = insertInd + probStart;
}


__kernel void gNthElement(
                __global float* restrict prob,
                uint rows, uint cols,
                __global uint* restrict beamSizes,
                uint beamSizesSize,
                __global uint* restrict d_cummulatedBeamSizes,
                __global uint* restrict d_batchFirstElementIdxs,
                uint maxBatchSize,
                __global float* restrict bestCost,
                __global unsigned* restrict bestInd
                )
{
  uint offset = 0;
  for (uint batchId = 0; batchId < beamSizesSize; ++batchId) {
    uint maxBeamSize = beamSizes[batchId];
    uint probStart = d_batchFirstElementIdxs[batchId];
    uint probEnd = d_batchFirstElementIdxs[batchId + 1];
    uint numElements = probEnd - probStart;
    
    //assert(rows == maxBatchSize);
    //assert(cols > maxBeamSize);
  
    // init arrays
    for (uint i = 0; i < maxBeamSize; ++i) {
      float val = prob[probStart + i];
      insertValue(bestCost + offset, bestInd + offset, i, val, i, probStart);
    }
  
    for (uint i = maxBeamSize; i < numElements; ++i) {
      float val = prob[probStart + i];
      replaceValueOrDiscard(bestCost + offset, bestInd + offset, maxBeamSize, val, i, probStart);
    }
    
    offset += maxBeamSize;
  }    
}



