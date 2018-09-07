add_subdirectory(3rd_party)

include_directories(.)
include_directories(3rd_party)
include_directories(3rd_party/SQLiteCpp/include)

add_library(marian STATIC
  common/utils.cpp
  common/logging.cpp
  common/config.cpp
  common/config_parser.cpp
  common/binary.cpp
  common/io.cpp

  data/alignment.cpp
  data/vocab.cpp
  data/corpus_base.cpp
  data/corpus.cpp
  data/corpus_sqlite.cpp
  data/corpus_nbest.cpp
  data/text_input.cpp

  3rd_party/cnpy/cnpy.cpp
  3rd_party/exception.cpp
  3rd_party/svd/svd.cpp

  tensors/backend.cpp
  tensors/cpu/device.cpp
  tensors/cpu/dropout.cpp
  tensors/cpu/prod.cpp
  tensors/cpu/tensor_operators.cpp

  tensors/cpu/sharp/avx_gemm.cpp
  tensors/cpu/sharp/sse_gemm.cpp

  graph/expression_graph.cpp
  graph/expression_operators.cpp
  graph/node.cpp
  graph/node_operators.cpp
  graph/node_initializers.cpp

  layers/convolution.cpp
  layers/loss.cpp
  layers/weight.cpp

  rnn/cells.cpp
  rnn/attention.cpp

  optimizers/clippers.cpp
  optimizers/optimizers.cpp

  models/model_factory.cpp
  models/encoder_decoder.cpp

  rescorer/score_collector.cpp

  translator/history.cpp
  translator/output_collector.cpp
  translator/output_printer.cpp
  translator/nth_element.cpp
  translator/helpers.cpp
  translator/scorers.cpp

  training/graph_group_async.cpp
  training/graph_group_async_drop.cpp
  training/graph_group_sync.cpp
  training/graph_group_singleton.cpp
  training/graph_group_multinode.cpp
  training/graph_group_multinode_sync.cpp
  training/validator.cpp
  training/communicator.cpp

  $<TARGET_OBJECTS:libyaml-cpp>
  $<TARGET_OBJECTS:SQLiteCpp>
)

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
  training/gradient_dropping/gpu/dropper.cu
  training/gradient_dropping/gpu/sparse_algorithm.cu
  training/communicator.cu
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

add_executable(marian_conv command/marian_conv.cpp)
set_target_properties(marian_conv PROPERTIES OUTPUT_NAME marian-conv)

set(EXECUTABLES ${EXECUTABLES} marian_train marian_decoder marian_scorer marian_vocab marian_conv)

# marian.zip and marian.tgz
# This combines marian, marian_decoder in a single ZIP or TAR file for
# execution in MSFT internal tools FLO and Philly.
# For Philly submission, we need statically-linked versions to deal with
# library dependencies, so this target is only enabled for static builds.
if(USE_STATIC_LIBS)
  add_custom_command(
    OUTPUT "${CMAKE_BINARY_DIR}/marian.zip"
    COMMAND zip -v -0 -j "${CMAKE_BINARY_DIR}/marian.zip"
                "${CMAKE_BINARY_DIR}/marian"
                "${CMAKE_BINARY_DIR}/marian-decoder"
                "${CMAKE_BINARY_DIR}/marian-scorer"
                "${CMAKE_BINARY_DIR}/marian-vocab"
                "${CMAKE_BINARY_DIR}/marian-conv"
    DEPENDS marian_train marian_decoder marian_scorer marian_vocab marian_conv)
  add_custom_target(marian_zip DEPENDS "${CMAKE_BINARY_DIR}/marian.zip")

  add_custom_command(
    OUTPUT "${CMAKE_BINARY_DIR}/marian.tgz"
    COMMAND tar -cvvzf "${CMAKE_BINARY_DIR}/marian.tgz" -C "${CMAKE_BINARY_DIR}"
                "marian"
                "marian-decoder"
                "marian-scorer"
                "marian-vocab"
                "marian-conv"
    DEPENDS marian_train marian_decoder marian_scorer marian_vocab marian_conv)
  add_custom_target(marian_tgz DEPENDS "${CMAKE_BINARY_DIR}/marian.tgz")

  add_custom_target(philly DEPENDS marian_tgz marian_zip)
endif()

if(COMPILE_SERVER)
  add_executable(marian_server command/marian_server.cpp)
  set_target_properties(marian_server PROPERTIES OUTPUT_NAME marian-server)
  set(EXECUTABLES ${EXECUTABLES} marian_server)
endif(COMPILE_SERVER)

foreach(exec ${EXECUTABLES})
  target_link_libraries(${exec} marian ${EXT_LIBS} ${EXT_LIBS} ${CMAKE_THREAD_LIBS_INIT})
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

if(COMPILE_TESTS)
  set(CATCH_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rd_party)
  add_library(Catch INTERFACE)
  target_include_directories(Catch INTERFACE ${CATCH_INCLUDE_DIR})

  add_subdirectory(tests)
endif(COMPILE_TESTS)

if(COMPILE_EXAMPLES)
  add_subdirectory(examples)
endif(COMPILE_EXAMPLES)
