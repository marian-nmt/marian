add_subdirectory(3rd_party)

include_directories(.)
include_directories(3rd_party)
include_directories(3rd_party/SQLiteCpp/include)

add_library(marian STATIC
  common/utils.cpp
  common/logging.cpp
  common/config.cpp
  common/config_parser.cpp

  data/vocab.cpp
  data/corpus_base.cpp
  data/corpus.cpp
  data/corpus_sqlite.cpp
  data/text_input.cpp

  3rd_party/cnpy/cnpy.cpp
  3rd_party/exception.cpp
  3rd_party/svd/svd.cpp

  tensors/backend.cpp
  tensors/cpu/device.cpp
  tensors/cpu/dropout.cpp
  tensors/cpu/prod.cpp
  tensors/cpu/tensor_operators.cpp

  graph/expression_graph.cpp
  graph/expression_operators.cpp
  graph/node.cpp
  graph/node_operators.cpp
  graph/node_initializers.cpp

  layers/convolution.cpp

  rnn/cells.cpp
  rnn/attention.cpp

  optimizers/clippers.cpp
  optimizers/optimizers.cpp

  models/model_factory.cpp

  translator/history.cpp
  translator/output_collector.cpp
  translator/nth_element.cpp
  translator/helpers.cpp
  translator/scorers.cpp

  training/graph_group_async.cpp
  training/graph_group_async_drop.cpp
  training/graph_group_sync.cpp
  training/graph_group_singleton.cpp
  training/graph_group_multinode.cpp
  training/validator.cpp

  rescorer/score_collector.cpp
  $<TARGET_OBJECTS:libyaml-cpp>
  $<TARGET_OBJECTS:SQLiteCpp>)

if(CUDA_FOUND)
cuda_add_library(marian_cuda
  tensors/gpu/device.cu
  tensors/gpu/algorithm.cu
  tensors/gpu/dropout.cu
  tensors/gpu/prod.cu
  tensors/gpu/element.cu
  tensors/gpu/add.cu
  tensors/gpu/tensor_operators.cu
  tensors/gpu/cudnn_wrappers.cu
  translator/nth_element.cu
  translator/helpers.cu
  training/dropper.cu
  training/sparse_tensor.cu
  STATIC)
endif(CUDA_FOUND)

set_target_properties(marian PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set_target_properties(marian PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

add_executable(marian_train command/marian.cpp)
set_target_properties(marian_train PROPERTIES OUTPUT_NAME marian)

add_executable(marian_decoder command/marian_decoder.cpp)
set_target_properties(marian_decoder PROPERTIES OUTPUT_NAME marian-decoder)

add_executable(marian_scorer command/marian_scorer.cpp)
set_target_properties(marian_scorer PROPERTIES OUTPUT_NAME marian-scorer)

add_executable(marian_vocab command/marian_vocab.cpp)
set_target_properties(marian_vocab PROPERTIES OUTPUT_NAME marian-vocab)

set(EXECUTABLES ${EXECUTABLES} marian_train marian_decoder marian_scorer marian_vocab)

if(COMPILE_SERVER)
  add_executable(marian_server command/marian_server.cpp)
  set_target_properties(marian_server PROPERTIES OUTPUT_NAME marian-server)
  set(EXECUTABLES ${EXECUTABLES} marian_server)
endif(COMPILE_SERVER)

foreach(exec ${EXECUTABLES})
  target_link_libraries(${exec} marian ${EXT_LIBS})
  if(CUDA_FOUND)
    target_link_libraries(${exec} marian marian_cuda ${EXT_LIBS} ${CMAKE_THREAD_LIBS_INIT})
    cuda_add_cublas_to_target(${exec})
  endif(CUDA_FOUND)
  set_target_properties(${exec} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endforeach(exec)

#add_executable(
#  align2steps
#  tools/align2steps.cpp
#)

#set_target_properties(align2steps PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

if(PYTHONLIBS_FOUND)
#  add_subdirectory(python)
endif(PYTHONLIBS_FOUND)

if(COMPILE_TESTS)
  set(CATCH_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rd_party)
  add_library(Catch INTERFACE)
  target_include_directories(Catch INTERFACE ${CATCH_INCLUDE_DIR})

  add_subdirectory(tests)
endif(COMPILE_TESTS)

if(COMPILE_EXAMPLES)
  add_subdirectory(examples)
endif(COMPILE_EXAMPLES)
