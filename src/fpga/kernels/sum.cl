__kernel void sum(                                                    
   __global float* input, 
   __global float* output,
   const unsigned int count)
{
  float ret = 0.0f;
  for (size_t i = 0; i < count; ++i) {
    ret += input[i];
  }
  (*output) = ret;
}                                      

