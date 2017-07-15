add_subdirectory(3rd_party)

include_directories(.)
include_directories(3rd_party)

cuda_add_library(marian
  3rd_party/cnpy/cnpy.cpp
  3rd_party/exception.cpp
  3rd_party/svd/svd.cpp
  graph/expression_graph.cu
  graph/expression_operators.cu
  graph/node.cu
  graph/node_operators.cu
  tensors/tensor.cu
  tensors/device_gpu.cu
  kernels/tensor_operators.cu
  kernels/dropout.cu
  kernels/sparse.cu
  layers/param_initializers.cu
  rnn/attention.cu
  rnn/cells.cu
  optimizers/clippers.cu
  optimizers/optimizers.cu
  common/utils.cpp
  common/logging.cpp
  common/config.cpp
  common/config_parser.cpp
  translator/history.cpp
  translator/output_collector.cpp
  translator/nth_element.cu
  translator/helpers.cu
  data/vocab.cpp
  data/corpus.cpp
  rescorer/score_collector.cpp
  $<TARGET_OBJECTS:libyaml-cpp>
  STATIC
)

set_target_properties(marian PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set_target_properties(marian PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

cuda_add_executable(marian_train command/marian.cu)
set_target_properties(marian_train PROPERTIES OUTPUT_NAME marian)
target_link_libraries(marian_train marian)

add_executable(marian_translate command/s2s_translator.cpp)
set_target_properties(marian_translate PROPERTIES OUTPUT_NAME s2s)
target_link_libraries(marian_translate marian)

add_executable(marian_rescorer command/s2s_rescorer.cpp)
set_target_properties(marian_rescorer PROPERTIES OUTPUT_NAME rescorer)
target_link_libraries(marian_rescorer marian)

foreach(exec marian_train marian_translate marian_rescorer)
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
  set(CATCH_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rd_party)
  add_library(Catch INTERFACE)
  target_include_directories(Catch INTERFACE ${CATCH_INCLUDE_DIR})

  add_subdirectory(tests)
endif(COMPILE_TESTS)

if(COMPILE_EXAMPLES)
  add_subdirectory(examples)
endif(COMPILE_EXAMPLES)
