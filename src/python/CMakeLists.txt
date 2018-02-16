cuda_add_library(pymarian SHARED
  mariannmt.cpp
  ../3rd_party/cnpy/cnpy.cpp
  ../3rd_party/exception.cpp
  ../3rd_party/svd/svd.cpp
  ../graph/expression_graph.cpp
  ../graph/expression_operators.cu
  ../graph/node.cu
  ../graph/node_operators.cu
  ../tensors/tensor.cu
  ../tensors/device.cpp
  ../kernels/tensor_operators.cu
  ../tensors/gpu/dropout.cu
  ../tensors/cpu/dropout.cpp
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
  ../data/text_input.cpp
  #../rescorer/score_collector.cpp
  $<TARGET_OBJECTS:libyaml-cpp>
)

set_target_properties(pymarian PROPERTIES EXCLUDE_FROM_ALL 1)
set_target_properties(pymarian PROPERTIES OUTPUT_NAME mariannmt)
set_target_properties(pymarian PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set_target_properties(pymarian PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

target_link_libraries(pymarian ${EXT_LIBS} marian)
cuda_add_cublas_to_target(pymarian)
