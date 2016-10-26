#include "gpu/decoder/nth_element.h"


namespace GPU {

static void HandleError(cudaError_t err, const char *file, int line ) {
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString( err ), file, line);
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
  __syncthreads();

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

  __syncthreads();

  for (int s = (blockDim.x >> 1); s > 0; s >>= 1) {
    if (tid < s) {
      if (sdata[tid + s] > sdata[tid]) {
        sdata[tid] = sdata[tid + s];
        indices[tid] = indices[tid + s];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    d_out[blockIdx.x] = sdata[0];
    d_ind[blockIdx.x] = indices[0];
  }
}

__global__ void gSet(float* d_in, int* d_idx, int* index) {
  *index = d_idx[*index];
  // d_in[*index] = std::numeric_limits<float>::lowest();
  d_in[*index] = -3.40282e+38f;
}

NthElement::NthElement(size_t maxBeamSize, cudaStream_t& stream)
    : stream_(stream) {
  HANDLE_ERROR( cudaMalloc((void**)&d_ind, BLOCK_SIZE * sizeof(int)) );

  HANDLE_ERROR( cudaMalloc((void**)&d_out, BLOCK_SIZE * sizeof(float)) );

  HANDLE_ERROR( cudaMalloc((void**)&d_res_idx, maxBeamSize * sizeof(int)) );
  HANDLE_ERROR( cudaMalloc((void**)&d_res, maxBeamSize * sizeof(float)) );

  cudaHostAlloc((void**) &h_res, maxBeamSize * sizeof(float), cudaHostAllocDefault);
  cudaHostAlloc((void**) &h_res_idx, maxBeamSize * sizeof(int), cudaHostAllocDefault);
}

void NthElement::getNBestList(float* d_in, size_t N, size_t n,
                              std::vector<unsigned>& outKeys,
                              std::vector<float>& outValues) {
  if (n == 0) return;

  const int N_BLOCKS = (N / (2 * BLOCK_SIZE)) + 1;
  /* std::cerr << "#BLOCKS: " << N_BLOCKS << std::endl; */
  /* cudaStreamSynchronize(stream_); */

  for (size_t i = 0; i < n; ++i) {
    gMaxElement<<<N_BLOCKS, BLOCK_SIZE, BLOCK_SIZE * sizeof(float), stream_>>>(d_out, d_ind, d_in, N);
    /* cudaStreamSynchronize(stream_); */
    /* float *tmp= new float[N_BLOCKS]; */
    /* cudaMemcpy(tmp, d_out, N_BLOCKS * sizeof(float), cudaMemcpyDeviceToHost); */
    /* for (int k = 0; k < N_BLOCKS; ++k) std::cerr << k << ": " << tmp[k] << "\t"; */
    /* std::cerr << std::endl; */
    /* delete [] tmp; */
    gMaxElement<<<1, (512 / 2), (512 /2 ) * sizeof(float), stream_>>>(d_res + i, d_res_idx + i, d_out, N_BLOCKS);
    gSet<<<1, 1, 0, stream_>>>(d_in, d_ind, d_res_idx + i);
  }

  HANDLE_ERROR( cudaMemcpyAsync(h_res, d_res, n * sizeof(float), cudaMemcpyDeviceToHost, stream_) );
  HANDLE_ERROR( cudaMemcpyAsync(h_res_idx, d_res_idx, n * sizeof(int), cudaMemcpyDeviceToHost, stream_) );

  cudaStreamSynchronize(stream_);

  for (size_t i = 0; i < n; ++i) {
    outKeys[i] = h_res_idx[i];
    outValues[i] = h_res[i];
    /* std::cerr << outKeys[i] << ": " << outValues[i] << "\t"; */
  }
  /* std::cerr << std::endl; */
}

}  // namespace GPU
