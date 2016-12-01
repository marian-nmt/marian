#include "gpu/mblas/nth_element.h"
#include <iostream>


namespace GPU {

static void HandleError(cudaError_t err, const char *file, int line ) {
  if (err != cudaSuccess) {
    std::cerr << "ERROR: " << cudaGetErrorString(err) << " in " << file << " at line " << line << std::endl;
    exit( EXIT_FAILURE );
  }
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

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

  if (tid < 32 && tid + 32 < in_size) {
    if (sdata[tid + 32] > sdata[tid]) {
      sdata[tid] = sdata[tid + 32];
      indices[tid] = indices[tid + 32];
    }
  }

  if (tid < 16 && tid + 16 < in_size) {
    if (sdata[tid + 16] > sdata[tid]) {
      sdata[tid] = sdata[tid + 16];
      indices[tid] = indices[tid + 16];
    }
  }

  if (tid < 8 && tid + 8 < in_size) {
    if (sdata[tid + 8] > sdata[tid]) {
      sdata[tid] = sdata[tid + 8];
      indices[tid] = indices[tid + 8];
    }
  }

  if (tid < 4 && tid + 4 < in_size) {
    if (sdata[tid + 4] > sdata[tid]) {
      sdata[tid] = sdata[tid + 4];
      indices[tid] = indices[tid + 4];
    }
  }

  if (tid < 2 && tid + 2 < in_size) {
    if (sdata[tid + 2] > sdata[tid]) {
      sdata[tid] = sdata[tid + 2];
      indices[tid] = indices[tid + 2];
    }
  }

  if (tid < 1 && tid + 1 < in_size) {
    if (sdata[tid + 1] > sdata[tid]) {
      sdata[tid] = sdata[tid + 1];
      indices[tid] = indices[tid + 1];
    }
  }

  if (tid == 0) {
    d_out[blockIdx.x] = sdata[0];
    d_ind[blockIdx.x] = indices[0];
  }
}

__global__ void gSet(float* d_in, int* d_idx, int* index) {
  *index = d_idx[*index];
  d_in[*index] = -3.40282e+38f;
}

__global__ void gMaxElementUpdate(float* d_out, int* d_ind, float* d_in, int* blockID, int dist, int in_size) {
  extern __shared__ float sdata[];

  int tid = threadIdx.x;
  if (tid == 0) {
    d_in[d_ind[*blockID]] = -3.40282e+38f;
  }

  __syncthreads();
  __shared__ int indices[512];

  int i = *blockID * (blockDim.x * 2) + tid;

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

  while (i + dist < in_size) {
    i += dist;

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
  i = *blockID * (blockDim.x * 2) + tid;

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

  if (tid < 32 && tid + 32 < in_size) {
    if (sdata[tid + 32] > sdata[tid]) {
      sdata[tid] = sdata[tid + 32];
      indices[tid] = indices[tid + 32];
    }
  }

  if (tid < 16 && tid + 16 < in_size) {
    if (sdata[tid + 16] > sdata[tid]) {
      sdata[tid] = sdata[tid + 16];
      indices[tid] = indices[tid + 16];
    }
  }

  if (tid < 8 && tid + 8 < in_size) {
    if (sdata[tid + 8] > sdata[tid]) {
      sdata[tid] = sdata[tid + 8];
      indices[tid] = indices[tid + 8];
    }
  }

  if (tid < 4 && tid + 4 < in_size) {
    if (sdata[tid + 4] > sdata[tid]) {
      sdata[tid] = sdata[tid + 4];
      indices[tid] = indices[tid + 4];
    }
  }

  if (tid < 2 && tid + 2 < in_size) {
    if (sdata[tid + 2] > sdata[tid]) {
      sdata[tid] = sdata[tid + 2];
      indices[tid] = indices[tid + 2];
    }
  }

  if (tid < 1 && tid + 1 < in_size) {
    if (sdata[tid + 1] > sdata[tid]) {
      sdata[tid] = sdata[tid + 1];
      indices[tid] = indices[tid + 1];
    }
  }

  if (tid == 0) {
    d_out[*blockID] = sdata[0];
    int tmp = d_ind[*blockID];
    d_ind[*blockID] = indices[0];
    *blockID = tmp;
  }
}
__global__ void gGetValueByKey(float* d_in, float* d_out, int* indeces, int n) {
  int tid = threadIdx.x  + blockDim.x * blockIdx.x;
  if (tid < n) {
    int index = indeces[tid];
    d_out[tid] = d_in[index];
  }
}

NthElement::NthElement(size_t maxBeamSize, size_t maxBatchSize, cudaStream_t& stream)
    : stream_(stream) {
  HANDLE_ERROR( cudaMalloc((void**)&d_ind, BLOCK_SIZE * sizeof(int)) );

  HANDLE_ERROR( cudaMalloc((void**)&d_out, BLOCK_SIZE * sizeof(float)) );

  HANDLE_ERROR( cudaMalloc((void**)&d_res_idx, maxBatchSize * maxBeamSize * sizeof(int)) );
  HANDLE_ERROR( cudaMalloc((void**)&d_res, maxBatchSize * maxBeamSize * sizeof(float)) );

  cudaHostAlloc((void**) &h_res, maxBeamSize * maxBatchSize* sizeof(float), cudaHostAllocDefault);
  cudaHostAlloc((void**) &h_res_idx, maxBeamSize * maxBatchSize * sizeof(int), cudaHostAllocDefault);

  HANDLE_ERROR( cudaMalloc((void**)&d_breakdown, maxBeamSize * sizeof(float)) );
}

void NthElement::getNBestList(float* d_in, size_t N, size_t n, size_t pos) {
  if (n == 0) return;

  const int N_BLOCKS = std::min(500, int(N / (2 * BLOCK_SIZE)) + int(N % (2 * BLOCK_SIZE) != 0));

  gMaxElement<<<N_BLOCKS, BLOCK_SIZE, BLOCK_SIZE * sizeof(float), stream_>>>
    (d_out, d_ind, d_in, N);

  for (size_t i = 0; i < n; ++i) {

    gMaxElement<<<1, 512, 512 * sizeof(float), stream_>>>
      (d_res + pos + i, d_res_idx + pos + i, d_out, N_BLOCKS);

    gMaxElementUpdate<<<1, BLOCK_SIZE, BLOCK_SIZE * sizeof(float), stream_>>>
      (d_out, d_ind, d_in, d_res_idx + pos + i, 2 * BLOCK_SIZE * N_BLOCKS, N);
  }
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
