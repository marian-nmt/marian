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
  kernels/dropout.cu
  layers/param_initializers.cpp
  common/utils.cpp
  common/logging.cpp
  common/history.cpp
  training/config.cpp
  translator/nth_element.cu
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
  marian_translate
  test/marian_translate.cu
)

cuda_add_executable(
  marian_test
  test/marian_test.cu
)

cuda_add_executable(
  bn_test
  test/bn_test.cu
)

cuda_add_executable(
  marian
  command/marian.cu
)

cuda_add_executable(
  dropout_test
  test/dropout_test.cu
)

target_link_libraries(marian marian_lib)
target_link_libraries(tensor_test marian_lib)
target_link_libraries(marian_test marian_lib)
target_link_libraries(dropout_test marian_lib)
target_link_libraries(marian_translate marian_lib)
target_link_libraries(bn_test marian_lib)

foreach(exec dropout_test tensor_test marian_test marian_translate marian bn_test)
  target_link_libraries(${exec} ${EXT_LIBS})
  cuda_add_cublas_to_target(${exec})
  set_target_properties(${exec} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endforeach(exec)
