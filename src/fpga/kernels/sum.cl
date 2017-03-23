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

__kernel void sum_size_t(                                                    
   __global size_t* input, 
   __global size_t* output,
   const unsigned int count)
{
  size_t ret = 0;
  for (size_t i = 0; i < count; ++i) {
    ret += input[i];
  }
  (*output) = ret;
}                                      

