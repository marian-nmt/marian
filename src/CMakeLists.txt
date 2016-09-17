
include_directories(.)

cuda_add_library(marian_lib 
  cnpy/cnpy.cpp
  exception.cpp
  expression_graph.cu 
  tensor.cu
  tensor_operators.cu
  expression_operators.cu
  vocab.cpp
)

target_link_libraries(marian_lib)

cuda_add_executable(
  marian
  test.cu
)

target_link_libraries(marian marian_lib)

cuda_add_executable(
  mnist_benchmark
  mnist_benchmark.cu
)

cuda_add_executable(
  validate_mnist_batch
  validate_mnist_batch.cu
)

cuda_add_executable(
  validate_encoder_decoder
  validate_encoder_decoder.cu
)

target_link_libraries(mnist_benchmark marian_lib)
target_link_libraries(validate_mnist_batch marian_lib)
target_link_libraries(validate_encoder_decoder marian_lib)

foreach(exec marian mnist_benchmark validate_mnist_batch validate_encoder_decoder)
  target_link_libraries(${exec} ${EXT_LIBS} cuda cudnn)
  cuda_add_cublas_to_target(${exec})
  set_target_properties(${exec} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endforeach(exec)
