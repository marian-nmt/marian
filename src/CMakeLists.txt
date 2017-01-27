add_subdirectory(3rd_party)

include_directories(.)
include_directories(3rd_party)

cuda_add_library(marian_lib
  3rd_party/cnpy/cnpy.cpp
  3rd_party/exception.cpp
  3rd_party/svd/svd.cpp
  graph/expression_graph.cu
  graph/expression_operators.cu
  graph/node.cu
  graph/node_operators.cu
  tensors/tensor.cu
  tensors/tensor_gpu.cu
  kernels/tensor_operators.cu
  layers/param_initializers.cpp
  common/utils.cpp
  $<TARGET_OBJECTS:libyaml-cpp>
)

target_link_libraries(marian_lib)

#cuda_add_executable(
#  softmax_benchmark
#  apps/softmax_benchmark.cu
#)

cuda_add_executable(
  tensor_test
  apps/tensor_test.cu
  data/vocab.cpp
  data/corpus.cpp
)

#cuda_add_executable(
#  mnist_benchmark
#  apps/mnist_benchmark.cu
#)

cuda_add_executable(
  nematus_test
  apps/nematus_test.cu
  data/vocab.cpp
  data/corpus.cpp
)

cuda_add_executable(
  nematus
  apps/nematus.cu
  data/vocab.cpp
  data/corpus.cpp
)

#target_link_libraries(softmax_benchmark marian_lib)
target_link_libraries(tensor_test marian_lib)
#target_link_libraries(mnist_benchmark marian_lib)
target_link_libraries(nematus_test marian_lib)
target_link_libraries(nematus marian_lib)

foreach(exec tensor_test nematus_test nematus)
  target_link_libraries(${exec} ${EXT_LIBS} cuda cudnn)
  cuda_add_cublas_to_target(${exec})
  set_target_properties(${exec} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endforeach(exec)
