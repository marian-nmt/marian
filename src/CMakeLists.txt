
include_directories(.)
include_directories(3rd_party)

cuda_add_library(marian_lib 
  3rd_party/cnpy/cnpy.cpp
  3rd_party/exception.cpp
  expression_graph.cu 
  expression_operators.cu
  node.cu
  node_operators_unary.cu
  node_operators_binary.cu
  tensor.cu
  tensor_operators.cu
  vocab.cpp
)

target_link_libraries(marian_lib)

cuda_add_executable(
  softmax_benchmark
  softmax_benchmark.cu
)

cuda_add_executable(
  mnist_benchmark
  mnist_benchmark.cu
)

#cuda_add_executable(
#  validate_encoder_decoder
#  validate_encoder_decoder.cu
#)

cuda_add_executable(
  test_nodes
  test_nodes.cu
)

target_link_libraries(softmax_benchmark marian_lib)
target_link_libraries(mnist_benchmark marian_lib)
#target_link_libraries(validate_encoder_decoder marian_lib)
target_link_libraries(test_nodes marian_lib)

foreach(exec mnist_benchmark softmax_benchmark test_nodes )
  target_link_libraries(${exec} ${EXT_LIBS} cuda cudnn curand)
  cuda_add_cublas_to_target(${exec})
  set_target_properties(${exec} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endforeach(exec)
