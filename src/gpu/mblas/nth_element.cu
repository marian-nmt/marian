#include "gpu/mblas/nth_element.h"
#include <iostream>


namespace GPU {

static void HandleError(cudaError_t err, const char *file, int line ) {
  if (err != cudaSuccess) {
    std::cerr << "ERROR: " << cudaGetErrorString(err) << " in " << file << " at line " << line << std::endl;
    exit( EXIT_FAILURE );
  }
}

#define UNROLL_MAXARG_LOOP( n, max ) \
  if (tid < (n) && tid + (n) < ( max ) ) { \
    if (sdata[tid + ( n ) ] > sdata[tid]) { \
      sdata[tid] = sdata[tid + ( n ) ]; \
      indices[tid] = indices[tid + ( n ) ]; \
    } \
  }

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

__global__ void gMaxElement(float* d_out, int* d_ind, float* d_in, int numBatches, int* batchFirstElementIdxs) {
  extern __shared__ float sdata[];
  __shared__ int indices[512];

  int tid = threadIdx.x;

  for (int batchIdx = 0; batchIdx < numBatches; ++batchIdx) {
    int begin = batchFirstElementIdxs[batchIdx];
    int end = batchFirstElementIdxs[batchIdx + 1];

    int i = begin + blockIdx.x * (blockDim.x * 2) + tid;

    sdata[tid] = -3.40282e+38f;

    /* if (i >= end) return; */
    if (i < end) {
      sdata[tid] = d_in[i];
      indices[tid] = i;
    }

    if (i + blockDim.x < end) {
      float a = d_in[i];
      float b = d_in[i + blockDim.x];
      if (a > b) {
        sdata[tid] = a;
        indices[tid] = i;
      } else {
        sdata[tid] = b;
        indices[tid] = i + blockDim.x;
      }
    }

    while (i + 2 * gridDim.x * blockDim.x < end) {
      i += 2 * gridDim.x * blockDim.x;

      float a = d_in[i];
      if (a > sdata[tid]) {
        sdata[tid] = a;
        indices[tid] = i;
      }

      if (i + blockDim.x < end) {
        float b = d_in[i + blockDim.x];
        if (b > sdata[tid]) {
          sdata[tid] = b;
          indices[tid] = i + blockDim.x;
        }
      }
    }

    __syncthreads();

    for (int s = (blockDim.x >> 1); s > 32; s >>= 1) {
      if (tid < s && tid + s < end) {
        if (sdata[tid + s] > sdata[tid]) {
          sdata[tid] = sdata[tid + s];
          indices[tid] = indices[tid + s];
        }
      }
      __syncthreads();
    }

    UNROLL_MAXARG_LOOP(32, end);
    UNROLL_MAXARG_LOOP(16, end);
    UNROLL_MAXARG_LOOP(8, end);
    UNROLL_MAXARG_LOOP(4, end);
    UNROLL_MAXARG_LOOP(2, end);
    UNROLL_MAXARG_LOOP(1, end);

    if (tid == 0) {
      d_out[blockIdx.x + batchIdx * gridDim.x] = sdata[0];
      d_ind[blockIdx.x + batchIdx * gridDim.x] = indices[0];
    }
  }
}

__global__ void gMaxElement(float* d_out, int* d_ind, float* d_in, int in_size) {
  extern __shared__ float sdata[];
  __shared__ int indices[512];


  int tid = threadIdx.x;
  int i = blockIdx.x * (blockDim.x * 2) + tid;

  sdata[tid] = -3.40282e+38f;

  if (i >= in_size) return;

  if (i + blockDim.x < in_size) {
    float a = d_in[i];
    float b = d_in[i+blockDim.x];
    if (a > b) {
      sdata[tid] = a;
      indices[tid] = i;
    } else {
      sdata[tid] = b;
      indices[tid] = i + blockDim.x;
    }
  } else {
    sdata[tid] = d_in[i];
    indices[tid] = i;
  }

  while (i + 2 * gridDim.x * blockDim.x < in_size) {
    i += 2 * gridDim.x * blockDim.x;

    float a = d_in[i];
    if (a > sdata[tid]) {
      sdata[tid] = a;
      indices[tid] = i;
    }

    if (i + blockDim.x < in_size) {
      float b = d_in[i + blockDim.x];
      if (b > sdata[tid]) {
        sdata[tid] = b;
        indices[tid] = i + blockDim.x;
      }
    }
  }
  i = blockIdx.x * (blockDim.x * 2) + tid;

  __syncthreads();

  for (int s = (blockDim.x >> 1); s > 32; s >>= 1) {
    if (tid < s && tid + s < in_size) {
      if (sdata[tid + s] > sdata[tid]) {
        sdata[tid] = sdata[tid + s];
        indices[tid] = indices[tid + s];
      }
    }
    __syncthreads();
  }

  UNROLL_MAXARG_LOOP(32, in_size);
  UNROLL_MAXARG_LOOP(16, in_size);
  UNROLL_MAXARG_LOOP(8, in_size);
  UNROLL_MAXARG_LOOP(4, in_size);
  UNROLL_MAXARG_LOOP(2, in_size);
  UNROLL_MAXARG_LOOP(1, in_size);

  if (tid == 0) {
    d_out[blockIdx.x] = sdata[0];
    d_ind[blockIdx.x] = indices[0];
  }
}

__global__ void gSet(float* d_in, int* d_idx, int* index) {
  *index = d_idx[*index];
  d_in[*index] = -3.40282e+38f;
}

__global__ void gMaxElementUpdate(float* binCosts, int* binIdxs, float* probs, int batchBeginIdx, int batchEndIdx, int num_bins, float* bestCost, int* blockID) {
  extern __shared__ float sdata[];
  __shared__ int indices[512];
  __shared__ float bestBinCost;
  __shared__ int bestBinCostIdx;

  const int tid = threadIdx.x;
  int i = blockIdx.x * (blockDim.x * 2) + tid;

  sdata[tid] = -3.40282e+38f;

  if (i < num_bins) {
    sdata[tid] = binCosts[i];
    indices[tid] = i;
  }

  if (i + blockDim.x < num_bins) {
    float a = binCosts[i];
    float b = binCosts[i+blockDim.x];
    if (a > b) {
      sdata[tid] = a;
      indices[tid] = i;
    } else {
      sdata[tid] = b;
      indices[tid] = i + blockDim.x;
    }
  }

  while (i + 2 * blockDim.x < num_bins) {
    i += 2 * blockDim.x;

    float a = binCosts[i];
    if (a > sdata[tid]) {
      sdata[tid] = a;
      indices[tid] = i;
    }

    if (i + blockDim.x < num_bins) {
      float b = binCosts[i + blockDim.x];
      if (b > sdata[tid]) {
        sdata[tid] = b;
        indices[tid] = i + blockDim.x;
      }
    }
  }

  __syncthreads();

  for (int s = (blockDim.x >> 1); s > 32; s >>= 1) {
    if (tid < s && tid + s < num_bins) {
      if (sdata[tid + s] > sdata[tid]) {
        sdata[tid] = sdata[tid + s];
        indices[tid] = indices[tid + s];
      }
    }
    __syncthreads();
  }

  UNROLL_MAXARG_LOOP(32, num_bins);
  UNROLL_MAXARG_LOOP(16, num_bins);
  UNROLL_MAXARG_LOOP(8, num_bins);
  UNROLL_MAXARG_LOOP(4, num_bins);
  UNROLL_MAXARG_LOOP(2, num_bins);
  UNROLL_MAXARG_LOOP(1, num_bins);

  if (tid == 0) {
    bestBinCost = sdata[0];
    bestBinCostIdx = indices[0];

    probs[binIdxs[bestBinCostIdx]] = -3.40282e+38f;

    *blockID = binIdxs[bestBinCostIdx];
    *bestCost = bestBinCost;
  }

  __syncthreads();

  i = batchBeginIdx + bestBinCostIdx * (blockDim.x * 2) + tid;
  const int dist = num_bins * 2 * blockDim.x;

  sdata[tid] = -3.40282e+38f;

  if (i < batchEndIdx) {
    sdata[tid] = probs[i];
    indices[tid] = i;
  }

  if (i + blockDim.x < batchEndIdx) {
    float a = probs[i];
    float b = probs[i+blockDim.x];
    if (a > b) {
      sdata[tid] = a;
      indices[tid] = i;
    } else {
      sdata[tid] = b;
      indices[tid] = i + blockDim.x;
    }
  }

  while (i + dist < batchEndIdx) {
    i += dist;

    float a = probs[i];
    if (a > sdata[tid]) {
      sdata[tid] = a;
      indices[tid] = i;
    }

    if (i + blockDim.x < batchEndIdx) {
      float b = probs[i + blockDim.x];
      if (b > sdata[tid]) {
        sdata[tid] = b;
        indices[tid] = i + blockDim.x;
      }
    }
  }

  __syncthreads();

  for (int s = (blockDim.x >> 1); s > 32; s >>= 1) {
    if (tid < s && tid + s < batchEndIdx) {
      if (sdata[tid + s] > sdata[tid]) {
        sdata[tid] = sdata[tid + s];
        indices[tid] = indices[tid + s];
      }
    }
    __syncthreads();
  }

  UNROLL_MAXARG_LOOP(32, batchEndIdx);
  UNROLL_MAXARG_LOOP(16, batchEndIdx);
  UNROLL_MAXARG_LOOP(8, batchEndIdx);
  UNROLL_MAXARG_LOOP(4, batchEndIdx);
  UNROLL_MAXARG_LOOP(2, batchEndIdx);
  UNROLL_MAXARG_LOOP(1, batchEndIdx);

  if (tid == 0) {
    binCosts[bestBinCostIdx] = sdata[0];
    binIdxs[bestBinCostIdx] = indices[0];
  }
}

__global__ void gGetValueByKey(float* d_in, float* d_out, int* indeces, int n)
{
  int tid = threadIdx.x  + blockDim.x * blockIdx.x;
  if (tid < n) {
    int index = indeces[tid];
    d_out[tid] = d_in[index];
  }
}

NthElement::NthElement(size_t maxBeamSize, size_t maxBatchSize, cudaStream_t& stream)
    : stream_(stream) ,
      NUM_BLOCKS(std::min(500, int(maxBeamSize * 85000 / (2 * BLOCK_SIZE)) + int(maxBeamSize * 85000 % (2 * BLOCK_SIZE) != 0)))
{
  HANDLE_ERROR( cudaMalloc((void**)&d_ind, maxBatchSize * NUM_BLOCKS * sizeof(int)) );

  HANDLE_ERROR( cudaMalloc((void**)&d_out, maxBatchSize * NUM_BLOCKS * sizeof(float)) );

  HANDLE_ERROR( cudaMalloc((void**)&d_res_idx, maxBatchSize * maxBeamSize * sizeof(int)) );
  HANDLE_ERROR( cudaMalloc((void**)&d_res, maxBatchSize * maxBeamSize * sizeof(float)) );

  HANDLE_ERROR( cudaHostAlloc((void**) &h_res, maxBeamSize * maxBatchSize* sizeof(float),
                              cudaHostAllocDefault) );
  HANDLE_ERROR( cudaHostAlloc((void**) &h_res_idx, maxBeamSize * maxBatchSize * sizeof(int),
                              cudaHostAllocDefault) );

  HANDLE_ERROR( cudaMalloc((void**)&d_breakdown, maxBeamSize * sizeof(float)) );
  HANDLE_ERROR( cudaMalloc((void**)&d_batchPosition, (maxBatchSize + 1) * sizeof(int)) );
}

void NthElement::getNBestList(float* probs, const std::vector<int>& batchFirstElementIdxs,
                              const std::vector<int>& cummulatedBeamSizes)
{
  HANDLE_ERROR( cudaMemcpyAsync(d_batchPosition, batchFirstElementIdxs.data(), batchFirstElementIdxs.size() * sizeof(int),
                                cudaMemcpyHostToDevice, stream_) );

  const int numBatches = batchFirstElementIdxs.size() - 1;

  /* std::cerr << "Computing maxing in buckets." << std::endl; */
  gMaxElement<<<NUM_BLOCKS, BLOCK_SIZE, BLOCK_SIZE * sizeof(float), stream_>>>
    (d_out, d_ind, probs, numBatches, d_batchPosition);

  /* cudaStreamSynchronize(stream_); */
  /* HANDLE_ERROR( cudaPeekAtLastError() ); */
  /* HANDLE_ERROR( cudaDeviceSynchronize() ); */

  /* int tmp[NUM_BLOCKS * numBatches]; */
  /* float tmpf[NUM_BLOCKS * numBatches]; */
  /* HANDLE_ERROR( cudaMemcpyAsync(tmp, d_ind, NUM_BLOCKS * numBatches * sizeof(int), */
                                /* cudaMemcpyDeviceToHost, stream_) ); */
  /* HANDLE_ERROR( cudaMemcpyAsync(tmpf, d_out, NUM_BLOCKS * numBatches * sizeof(float), */
                                /* cudaMemcpyDeviceToHost, stream_) ); */
  /* cudaStreamSynchronize(stream_); */
  /* HANDLE_ERROR( cudaPeekAtLastError() ); */
  /* HANDLE_ERROR( cudaDeviceSynchronize() ); */
  /* for (int k = 0; k < numBatches; ++k) { */
    /* for (int i = 0; i < NUM_BLOCKS; ++i) std::cerr << tmp[k * NUM_BLOCKS + i] << ":" << tmpf[k * NUM_BLOCKS + i] << "\t"; */
    /* std::cerr << std::endl; */
  /* } */

  for (int batchIdx = 0; batchIdx < numBatches; ++batchIdx) {
    const int N = batchFirstElementIdxs[batchIdx + 1] - batchFirstElementIdxs[batchIdx];
    const int NUM_BUCKETS(std::min(500, int(N / (2 * BLOCK_SIZE)) + int(N % (2 * BLOCK_SIZE) != 0)));
    /* std::cerr << "batch idx: " << batchIdx << std::endl; */
    /* std::cerr << "N: " << N << std::endl; */
    /* std::cerr << "NUM_BLOCKS: " << NUM_BLOCKS << std::endl; */
    /* std::cerr << "BLOCK_SIZE: " << BLOCK_SIZE << std::endl; */
    /* std::cerr << "NUM_BUCKETS: " << NUM_BUCKETS << std::endl; */

    for (size_t pos = cummulatedBeamSizes[batchIdx]; pos < cummulatedBeamSizes[batchIdx + 1]; ++pos) {
      gMaxElementUpdate<<<1, BLOCK_SIZE, BLOCK_SIZE * sizeof(float), stream_>>>
        (d_out + batchIdx * NUM_BLOCKS,
         d_ind + batchIdx * NUM_BLOCKS,
         probs,
         batchFirstElementIdxs[batchIdx],
         batchFirstElementIdxs[batchIdx + 1],
         NUM_BLOCKS,
         d_res + pos,
         d_res_idx + pos);

      /* cudaStreamSynchronize(stream_); */
      /* HANDLE_ERROR( cudaPeekAtLastError() ); */
      /* HANDLE_ERROR( cudaDeviceSynchronize() ); */

      /* int tt; */
      /* HANDLE_ERROR( cudaMemcpyAsync(&tt, d_res_idx + pos, sizeof(int), */
                                    /* cudaMemcpyDeviceToHost, stream_) ); */
      /* cudaStreamSynchronize(stream_); */
      /* HANDLE_ERROR( cudaPeekAtLastError() ); */
      /* HANDLE_ERROR( cudaDeviceSynchronize() ); */

      /* std::cerr << "TT: " << tt << std::endl; */
    }
  }
}

void NthElement::getNBestList(const std::vector<size_t>& beamSizes, mblas::Matrix& Probs,
                  std::vector<float>& outCosts, std::vector<unsigned>& outKeys,
                  const bool isFirst) {
  std::vector<int> cummulatedBeamSizes(beamSizes.size() + 1, 0);
  std::vector<int> batchFirstElementIdxs(beamSizes.size() + 1, 0);

  const size_t vocabSize = Probs.Cols();
  for (size_t i = 0; i < beamSizes.size(); ++i) {

    cummulatedBeamSizes[i + 1] = cummulatedBeamSizes[i] + beamSizes[i];
    batchFirstElementIdxs[i + 1] += ((isFirst) ? (i + 1) : cummulatedBeamSizes[i + 1]) * vocabSize;
    /* std::cerr << "BEAM: " << beamSizes[i] << "\tCUM: " << cummulatedBeamSizes[i + 1] << "\tFIRST: " << batchFirstElementIdxs[i + 1] << std::endl; */
  }

  getNBestList(Probs.data(), batchFirstElementIdxs, cummulatedBeamSizes);
  GetPairs(cummulatedBeamSizes.back(), outKeys, outCosts);

  /* for (size_t i = 0; i < cummulatedBeamSizes.back(); ++i) { */
    /* std::cerr << i << " " << outKeys[i] << " " << outCosts[i] << std::endl; */
  /* } */

  /* for (size_t batchIdx = 0; batchIdx < beamSizes.size(); ++batchIdx) { */
    /* for (int i = cummulatedBeamSizes[batchIdx]; i < cummulatedBeamSizes[batchIdx + 1]; ++i) { */
      /* outKeys[i] += batchFirstElementIdxs[batchIdx]; */
    /* } */
  /* } */
}

void NthElement::GetPairs(size_t number,
                    std::vector<unsigned>& outKeys,
                    std::vector<float>& outValues) {

  HANDLE_ERROR( cudaMemcpyAsync(h_res, d_res, number * sizeof(float),
                                cudaMemcpyDeviceToHost, stream_) );
  HANDLE_ERROR( cudaMemcpyAsync(h_res_idx, d_res_idx, number * sizeof(int),
                                cudaMemcpyDeviceToHost, stream_) );
  cudaStreamSynchronize(stream_);

  for (size_t i = 0; i < number; ++i) {
    outKeys.push_back(h_res_idx[i]);
    outValues.push_back(h_res[i]);
  }

  lastN = number;
}

void NthElement::getValueByKey(std::vector<float>& out, float* d_in) {
  gGetValueByKey<<<1, lastN, 0, stream_>>>
    (d_in, d_breakdown, h_res_idx, lastN);

  HANDLE_ERROR( cudaMemcpyAsync(out.data(), d_breakdown, lastN * sizeof(float),
                                cudaMemcpyDeviceToHost, stream_) );
  cudaStreamSynchronize(stream_);
}

}  // namespace GPU
