#include <iostream>
#include <vector>

using namespace std;

int main()
{
  cerr << "Starting" << endl;

  int NUM = 100;
  vector<float> h_vec1(NUM);
  vector<float> h_vec2(NUM);
  
  for (size_t i = 0; i < NUM; ++i) {
    h_vec1[i] = i * 3;
  }

  int *d_vec;
  cudaMalloc(&d_vec, NUM * sizeof(float));

  // copy
  //cudaMemcpy(d_vec, h_vec1.data(), NUM * sizeof(float), cudaMemcpyHostToDevice);
  //cudaMemcpy(h_vec2.data(), d_vec, NUM * sizeof(float), cudaMemcpyDeviceToHost);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  cudaMemcpyAsync(d_vec, h_vec1.data(), NUM * sizeof(float), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(h_vec2.data(), d_vec, NUM * sizeof(float), cudaMemcpyDeviceToHost, stream);

  cerr << "h_vec2=";
  for (size_t i = 0; i < NUM; ++i) {
    cerr << h_vec2[i] << " ";
  }
  cerr << endl;

  cudaStreamDestroy(stream);

  cerr << "Finished" << endl;
}


