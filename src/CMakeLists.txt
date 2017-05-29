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
  kernels/sparse.cu
  layers/param_initializers.cpp
  layers/attention.cu
  common/utils.cpp
  common/logging.cpp
  training/config.cpp
  translator/history.cpp
  translator/output_collector.cpp
  translator/nth_element.cu
  translator/helpers.cu
  data/vocab.cpp
  data/corpus.cpp
  $<TARGET_OBJECTS:libyaml-cpp>
)

target_link_libraries(marian_lib)

cuda_add_executable(marian_train command/marian.cu)
set_target_properties(marian_train PROPERTIES OUTPUT_NAME marian)
target_link_libraries(marian_train marian_lib)

cuda_add_executable(marian_translate command/s2s_translator.cu)
set_target_properties(marian_translate PROPERTIES OUTPUT_NAME s2s)
target_link_libraries(marian_translate marian_lib)



foreach(exec marian_train marian_translate)
  target_link_libraries(${exec} ${EXT_LIBS})
  cuda_add_cublas_to_target(${exec})
  set_target_properties(${exec} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endforeach(exec)

#add_executable(
#  align2steps
#  tools/align2steps.cpp
#)

#set_target_properties(align2steps PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

if(COMPILE_TESTS)
cuda_add_executable(
  tensor_test
  test/tensor_test.cu
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
  dropout_test
  test/dropout_test.cu
)

add_executable(
  logger_test
  test/logger_test.cpp
)

target_link_libraries(tensor_test marian_lib)
target_link_libraries(marian_test marian_lib)
target_link_libraries(dropout_test marian_lib)
target_link_libraries(bn_test marian_lib)

foreach(exec logger_test dropout_test tensor_test marian_test bn_test)
  target_link_libraries(${exec} ${EXT_LIBS})
  cuda_add_cublas_to_target(${exec})
  set_target_properties(${exec} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endforeach(exec)
endif(COMPILE_TESTS)

if(COMPILE_EXAMPLES)
  add_subdirectory(examples)
endif(COMPILE_EXAMPLES)
