add_subdirectory(3rd_party)

include_directories(.)
include_directories(3rd_party)

cuda_add_library(marian_lib
  3rd_party/cnpy/cnpy.cpp
  3rd_party/exception.cpp
  expression_graph.cu
  expression_operators.cu
  node.cu
  node_operators.cu
  node_operators_unary.cu
  node_operators_binary.cu
  tensors/tensor.cu
  tensors/tensor_gpu.cu
  tensor_operators.cu
  param_initializers.cpp
  common/vocab.cpp
  common/utils.cpp
  $<TARGET_OBJECTS:libyaml-cpp>
)

target_link_libraries(marian_lib)

cuda_add_executable(
  softmax_benchmark
  softmax_benchmark.cu
)

cuda_add_executable(
  tensor_test
  tensor_test.cu
)

cuda_add_executable(
  mnist_benchmark
  mnist_benchmark.cu
)

cuda_add_executable(
  nematus_test
  nematus_test.cu
)

target_link_libraries(softmax_benchmark marian_lib)
target_link_libraries(tensor_test marian_lib)
target_link_libraries(mnist_benchmark marian_lib)
target_link_libraries(nematus_test marian_lib)

foreach(exec mnist_benchmark tensor_test softmax_benchmark nematus_test)
  target_link_libraries(${exec} ${EXT_LIBS} cuda cudnn curand)
  cuda_add_cublas_to_target(${exec})
  set_target_properties(${exec} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endforeach(exec)
