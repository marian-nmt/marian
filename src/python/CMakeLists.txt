cuda_add_library(python SHARED
  mariannmt.cpp
  ../3rd_party/cnpy/cnpy.cpp
  ../3rd_party/exception.cpp
  ../3rd_party/svd/svd.cpp
  ../graph/expression_graph.cu
  ../graph/expression_operators.cu
  ../graph/node.cu
  ../graph/node_operators.cu
  ../tensors/tensor.cu
  ../tensors/device_gpu.cu
  ../kernels/tensor_operators.cu
  ../kernels/dropout.cu
  ../kernels/sparse.cu
  #../layers/param_initializers.cu
  ../rnn/attention.cu
  ../rnn/cells.cu
  #../optimizers/clippers.cu
  #../optimizers/optimizers.cu
  ../common/utils.cpp
  ../common/logging.cpp
  ../common/config.cpp
  ../common/config_parser.cpp
  ../translator/history.cpp
  ../translator/output_collector.cpp
  ../translator/nth_element.cu
  ../translator/helpers.cu
  ../data/vocab.cpp
  ../data/corpus.cpp
  #../rescorer/score_collector.cpp
  $<TARGET_OBJECTS:libyaml-cpp>
)

set_target_properties(python PROPERTIES EXCLUDE_FROM_ALL 1)
set_target_properties(python PROPERTIES OUTPUT_NAME mariannmt)
set_target_properties(python PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set_target_properties(python PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

target_link_libraries(python ${EXT_LIBS} marian)
cuda_add_cublas_to_target(python)
