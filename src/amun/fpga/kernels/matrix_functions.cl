
/////////////////////////////////////////////////////////////////////////////

float sumBase(__global float *input, unsigned count)
{
  float ret = 0.0f;
  for (unsigned i = 0; i < count; ++i) {
    ret += input[i];
  }
  return ret;
}

__kernel void sum_float(                                                    
   __global float* restrict input, 
   __global float* restrict output,
   const unsigned count)
{
  float ret = sumBase(input, count);
  (*output) = ret;
}                                      

__kernel void sum_uint(                                                    
   __global unsigned* restrict input, 
   __global unsigned* restrict output,
   const unsigned count)
{
  unsigned ret = 0;
  for (unsigned i = 0; i < count; ++i) {
    ret += input[i];
  }
  (*output) = ret;
}                                      

/////////////////////////////////////////////////////////////////////////////

__kernel void gCopyRows(
	__global float* restrict out, 
	__global const float* restrict in, 
	const unsigned cols,
  __global const unsigned* restrict targetRowIdx,
  const unsigned numPairs) 
{
  for (unsigned j = 0; j < numPairs; ++j) {
    unsigned srcId = targetRowIdx[j];    
    __global float *rowOut = out + j * cols;

    unsigned inOffset =  srcId * cols;
    __global const float *rowIn = in + inOffset;
    
  	for (unsigned i = 0; i < cols; ++i) {
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
  const unsigned rows,
  const unsigned cols)
{
  unsigned i = 0;
  for (unsigned row = 0; row < rows; ++row) {
    for (unsigned col = 0; col < cols; ++col) {
      float v = in[i];
      
      //unsigned outInd = row * cols + col;
      unsigned outInd = col * rows + row;
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
  const unsigned rowsA,
  const unsigned colsA,
  const unsigned rowsB,
  const unsigned colsB)
{
  for (unsigned rowA = 0; rowA < rowsA; ++rowA) {
    for (unsigned colB = 0; colB < colsB; ++colB) {
      float sum = 0;
      
      for (unsigned colA = 0; colA < colsA; ++colA) {
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
                                unsigned rows, unsigned cols) 
{
}

/////////////////////////////////////////////////////////////////////////////

__kernel void gElementTanh(__global float* restrict out, 
                          __global const float* restrict in1, 
                          __global const float* restrict in2,
                          unsigned rows, unsigned cols) 
{
  unsigned noElements = rows * cols;
  for (unsigned i = 0; i < noElements; ++i) {
    out[i] = tanh(out[i] + in1[i] * in2[i]);
  }
}

/////////////////////////////////////////////////////////////////////////////

__kernel void gElementTanh2(__global float* restrict out, 
                          __global const float* restrict in1, 
                          __global const float* restrict in2,
                          unsigned rows, unsigned cols) 
{
  unsigned noElements = rows * cols;
  for (unsigned i = 0; i < noElements; ++i) {
    out[i] = tanh(out[i] + in1[i] + in2[i]);
  }
}

/////////////////////////////////////////////////////////////////////////////

__kernel void gElementWhatever(__global float* restrict out, 
                          __global const float* restrict in1, 
                          __global const float* restrict in2,
                          unsigned rows, unsigned cols) 
{
  unsigned noElements = rows * cols;
  for (unsigned i = 0; i < noElements; ++i) {
    // (1.0 - _1) * _2 + _1 * _3
    out[i] = (1.0f - out[i]) * in1[i] + out[i] * in2[i];
  }
}

/////////////////////////////////////////////////////////////////////////////

__kernel void gElementAddWeighted(__global float* restrict out, 
                          __global const float* restrict in, 
                          unsigned rows, unsigned cols, float weight) 
{
  unsigned noElements = rows * cols;
  for (unsigned i = 0; i < noElements; ++i) {
    // _1 + weights_.at(scorers[i]->GetName()) * _2
    out[i] = out[i] + weight * in[i];
  }
}

/////////////////////////////////////////////////////////////////////////////

__kernel void gBroadcastVecAdd(__global float* restrict out, 
                              __global const float* restrict in, 
                              unsigned rows, unsigned cols) 
{
  for (unsigned noColumn = 0; noColumn < cols; ++noColumn) {
    float vecValue = in[noColumn];
  
    unsigned index = noColumn;
    for (unsigned noRow = 0; noRow < rows; ++noRow) {
        out[index] += vecValue;
        index += cols;
    }
  
  }

}

/////////////////////////////////////////////////////////////////////////////

__kernel void gBroadcastVecTanh(__global float* restrict out, 
                              __global const float* restrict in, 
                              unsigned rows, unsigned cols) 
{
  for (unsigned noColumn = 0; noColumn < cols; ++noColumn) {
    float vecValue = in[noColumn];
  
    unsigned index = noColumn;
    for (unsigned noRow = 0; noRow < rows; ++noRow) {
        out[index] = tanh(out[index] + vecValue);
        index += cols;
    }
  
  }

}

/////////////////////////////////////////////////////////////////////////////

__kernel void gBroadcastTanh(__global float* restrict out, 
                            __global const float* restrict in1, 
                            __global const float* restrict in2,
                            unsigned srcSize, 
                            unsigned cols, 
                            __global const int* restrict batchMapping,
                           unsigned batchMappingSize, 
                           unsigned outSize, 
                           unsigned in1Size, 
                           unsigned in2Size,
                           unsigned inRows)
{
  unsigned maxId = srcSize * inRows * cols;
  for (unsigned id = 0; id < maxId; ++id) {
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
                     unsigned rows, unsigned cols,
                     float weight) 
{
  for (unsigned noColumn = 0; noColumn < cols; ++noColumn) {
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
                     unsigned rows, unsigned cols) 
{
  unsigned i = 0;
  
  for (unsigned noColumn = 0; noColumn < cols; ++noColumn) {
    for (unsigned noRow = 0; noRow < rows; ++noRow) {
      float p = out[i] + in[i];
      out[i] = 1.0f / (1.0f + exp(-p));
      ++i;
    }
  }  
}  

/////////////////////////////////////////////////////////////////////////////

__kernel void gSlice(__global float* restrict out, 
                    __global const float* restrict in,
                   unsigned n, unsigned dim,
                   unsigned rows, unsigned cols) 
{
  unsigned offset = n * dim;

  for (unsigned row = 0; row < rows; ++row) {
    for (unsigned outCol = 0; outCol < dim; ++outCol) {
      unsigned inCol = offset + outCol;
      
      unsigned inInd = row * cols + inCol;
      unsigned outInd = row * dim + outCol;
      
      out[outInd] = in[inInd];
    }
  }
}

/////////////////////////////////////////////////////////////////////////////

__kernel void gPasteRows(__global float* restrict d_out, 
                    unsigned outRows, 
                    unsigned outCols, 
                    __global const float* restrict d_in, 
                    unsigned inRows, 
                    unsigned inCols, 
                    unsigned colNo, 
                    unsigned sparse) 
{
  unsigned maxId = inRows * inCols;
  for (unsigned id = 0; id < maxId; ++id) {
    unsigned inRow = id / inCols;
    unsigned inCol = id % inCols;
    unsigned outID = (outRows + sparse * inRow) * outCols + inCol + colNo;
    d_out[outID] = d_in[id];
  }
}

/////////////////////////////////////////////////////////////////////////////

__kernel void gMapMatrix(__global float* restrict d_in, 
                    unsigned numRows, 
                    unsigned numCols, 
                    unsigned mappingCols, 
                    __global const int* restrict mapping, 
                    unsigned i) 
{
  for (unsigned batchIdx = 0; batchIdx < numRows; ++batchIdx) {
    int isWord = mapping[mappingCols * batchIdx + i];
    if (isWord == 0) {
      // blank out word
      for (unsigned j = 0; j < numCols; ++j) {
        unsigned ind = batchIdx * numCols + j;
        d_in[ind] = 0;
      }
    }
  }
}

/////////////////////////////////////////////////////////////////////////////

__kernel void gMean(__global float* restrict d_out, 
                    __global const float* restrict d_in, 
                    __global const int* restrict mapping,
                    unsigned batchNum, unsigned senLen, unsigned stateLength) 
{
  for (unsigned id = 0; id < stateLength; ++id) {
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
                       unsigned rows, 
                       unsigned cols,
                       __global const int* restrict batchIds,
                       unsigned batchNum,
                       __global const int* restrict srcMapping,
                       unsigned srcNum) 
{
  for (unsigned row = 0; row < rows; ++row) {
    unsigned indRow = row * cols;
    
    int batchId = batchIds[row];
    unsigned startBatchInd = batchId * srcNum;
     
    // EXP
    float sumExp = 0;
    for (unsigned col = 0; col < cols; ++col) {
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
    for (unsigned col = 0; col < cols; ++col) {
      softMaxP[indRow + col] /= sumExp;
    }    
  }
}                         

/////////////////////////////////////////////////////////////////////////////

__kernel void gLogSoftMax(__global float* restrict softMaxP, 
                       unsigned rows, 
                       unsigned cols)
{
  // probably only work for non-batch
  for (unsigned row = 0; row < rows; ++row) {
    unsigned indRow = row * cols;

    // EXP
    float sumExp = 0;
    for (unsigned col = 0; col < cols; ++col) {
      float val = softMaxP[indRow + col];
      val = exp(val);
      
      sumExp += val;
      softMaxP[indRow + col] = val;
    }
    
    // NORMALIZE
    for (unsigned col = 0; col < cols; ++col) {
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
                            unsigned numRows, unsigned numCols, unsigned srcLen) 
{

  unsigned size = numRows * numCols;
  for (unsigned id = 0; id < size; ++id) {
    unsigned rowNo = id / numCols;
    unsigned batchNo = mapping[rowNo];
    unsigned statePos = id % numCols;
  
    float sum = 0.0f;
    for (unsigned i = 0; i < srcLen; ++i) {
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
                unsigned count,
                float val,
                unsigned insertInd,
                unsigned probStart)
{
  unsigned ind = count;
  for (unsigned i = 1; i < count; ++i) {
    if (val <= bestCost[i]) {
      ind = i;
      break;
    }
  }
  
  // shift everything up 1
  for (unsigned i = ind; i < count; ++i) {
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
                unsigned count,
                float val,
                unsigned insertInd,
                unsigned probStart)
{
  if (val < bestCost[0]) {
    // too low
    return;
  }
  
  unsigned ind = count - 1;
  for (unsigned i = 1; i < count; ++i) {
    if (val <= bestCost[i]) {
      ind = i - 1;
      break;
    }
  }
  
  // shift lowest value out of the array
  for (unsigned i = 0; i < ind; ++i) {
    bestCost[i] = bestCost[i+1];
    bestInd[i] = bestInd[i+1];
  }
  
  // insert value into place
  bestCost[ind] = val;
  bestInd[ind] = insertInd + probStart;
}


__kernel void gNthElement(
                __global float* restrict prob,
                unsigned rows, unsigned cols,
                __global unsigned* restrict beamSizes,
                unsigned beamSizesSize,
                __global unsigned* restrict d_cummulatedBeamSizes,
                __global unsigned* restrict d_batchFirstElementIdxs,
                unsigned maxBatchSize,
                __global float* restrict bestCost,
                __global unsigned* restrict bestInd
                )
{
  unsigned offset = 0;
  for (unsigned batchId = 0; batchId < beamSizesSize; ++batchId) {
    unsigned maxBeamSize = beamSizes[batchId];
    unsigned probStart = d_batchFirstElementIdxs[batchId];
    unsigned probEnd = d_batchFirstElementIdxs[batchId + 1];
    unsigned numElements = probEnd - probStart;
    
    //assert(rows == maxBatchSize);
    //assert(cols > maxBeamSize);
  
    // init arrays
    for (unsigned i = 0; i < maxBeamSize; ++i) {
      float val = prob[probStart + i];
      insertValue(bestCost + offset, bestInd + offset, i, val, i, probStart);
    }
  
    for (unsigned i = maxBeamSize; i < numElements; ++i) {
      float val = prob[probStart + i];
      replaceValueOrDiscard(bestCost + offset, bestInd + offset, maxBeamSize, val, i, probStart);
    }
    
    offset += maxBeamSize;
  }    
}



