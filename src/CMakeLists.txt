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
  kernels/tensor_operators.cu
  layers/param_initializers.cpp
  common/utils.cpp
  data/vocab.cpp
  data/corpus.cpp
  $<TARGET_OBJECTS:libyaml-cpp>
)

target_link_libraries(marian_lib)

cuda_add_executable(
  tensor_test
  test/tensor_test.cu
)

cuda_add_executable(
  marian_test
  test/marian_test.cu
)

cuda_add_executable(
  marian
  command/config.cpp
  command/marian.cu
)

cuda_add_executable(
  dropout_test
  test/dropout_test.cu
  kernels/dropout_cudnn.cu
)

target_link_libraries(marian marian_lib)
target_link_libraries(tensor_test marian_lib)
target_link_libraries(marian_test marian_lib)
target_link_libraries(dropout_test marian_lib)

foreach(exec tensor_test marian_test marian dropout_test)
  target_link_libraries(${exec} ${EXT_LIBS} cudnn)
  target_link_libraries(${exec} ${EXT_LIBS} curand)
  cuda_add_cublas_to_target(${exec})
  set_target_properties(${exec} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endforeach(exec)
