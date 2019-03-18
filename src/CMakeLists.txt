add_subdirectory(3rd_party)

include_directories(.)
include_directories(3rd_party)
include_directories(3rd_party/SQLiteCpp/include)
include_directories(3rd_party/sentencepiece)
include_directories(${CMAKE_BINARY_DIR}/local/include)

add_library(marian STATIC
  common/version.cpp
  common/utils.cpp
  common/logging.cpp
  common/cli_helper.cpp
  common/cli_wrapper.cpp
  common/config.cpp
  common/config_parser.cpp
  common/config_validator.cpp
  common/binary.cpp
  common/io.cpp

  data/alignment.cpp
  data/vocab.cpp
  data/default_vocab.cpp
  data/sentencepiece_vocab.cpp
  data/corpus_base.cpp
  data/corpus.cpp
  data/corpus_sqlite.cpp
  data/corpus_nbest.cpp
  data/text_input.cpp

  3rd_party/cnpy/cnpy.cpp
  3rd_party/ExceptionWithCallStack.cpp

  tensors/backend.cpp
  tensors/rand.cpp
  tensors/cpu/device.cpp
  tensors/cpu/prod.cpp
  tensors/cpu/tensor_operators.cpp

  tensors/cpu/sharp/int_gemm.cpp
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
  models/transformer_stub.cpp

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

  # this is only compiled to catch build errors, but not linked
  microsoft/quicksand.cpp

  $<TARGET_OBJECTS:libyaml-cpp>
  $<TARGET_OBJECTS:SQLiteCpp>
  $<TARGET_OBJECTS:pathie-cpp>
  $<TARGET_OBJECTS:zlib>
)
target_compile_options(marian PUBLIC ${ALL_WARNINGS})

# Generate git_revision.h to reflect current git revision information
# [https://stackoverflow.com/questions/1435953/how-can-i-pass-git-sha1-to-compiler-as-definition-using-cmake]
# Git updates .git/logs/HEAD file whenever you pull or commit something.
add_custom_command(OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/common/git_revision.h
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMAND git log -1 --pretty=format:\#define\ GIT_REVISION\ \"\%h\ \%ai\" > ${CMAKE_CURRENT_SOURCE_DIR}/common/git_revision.h
  DEPENDS ${CMAKE_SOURCE_DIR}/.git/logs/HEAD
  VERBATIM
)
add_custom_target(marian_version DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/common/git_revision.h)
add_dependencies(marian marian_version) # marian must depend on it so that it gets created first
# make sure all local dependencies are installed first before this is built
add_dependencies(marian 3rd_party_installs)

if(CUDA_FOUND)
cuda_add_library(marian_cuda
  tensors/gpu/device.cu
  tensors/gpu/algorithm.cu
  tensors/gpu/prod.cpp
  tensors/gpu/element.cu
  tensors/gpu/add.cu
  tensors/gpu/tensor_operators.cu
  tensors/gpu/cudnn_wrappers.cu
  translator/nth_element.cu
  translator/helpers.cu
  training/gradient_dropping/gpu/dropper.cu
  training/gradient_dropping/gpu/sparse_algorithm.cu
  STATIC)

  target_compile_options(marian_cuda PUBLIC ${ALL_WARNINGS})
  # make sure all local dependencies are installed first before this is built
  add_dependencies(marian_cuda 3rd_party_installs)
endif(CUDA_FOUND)

set_target_properties(marian PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set_target_properties(marian PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

add_executable(marian_train command/marian_main.cpp)
set_target_properties(marian_train PROPERTIES OUTPUT_NAME marian)
target_compile_options(marian_train PUBLIC ${ALL_WARNINGS})

add_executable(marian_decoder command/marian_decoder.cpp)
set_target_properties(marian_decoder PROPERTIES OUTPUT_NAME marian-decoder)
target_compile_options(marian_decoder PUBLIC ${ALL_WARNINGS})

add_executable(marian_scorer command/marian_scorer.cpp)
set_target_properties(marian_scorer PROPERTIES OUTPUT_NAME marian-scorer)
target_compile_options(marian_scorer PUBLIC ${ALL_WARNINGS})

add_executable(marian_vocab command/marian_vocab.cpp)
set_target_properties(marian_vocab PROPERTIES OUTPUT_NAME marian-vocab)
target_compile_options(marian_vocab PUBLIC ${ALL_WARNINGS})

add_executable(marian_conv command/marian_conv.cpp)
set_target_properties(marian_conv PROPERTIES OUTPUT_NAME marian-conv)
target_compile_options(marian_conv PUBLIC ${ALL_WARNINGS})

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
endif(USE_STATIC_LIBS)

if(COMPILE_SERVER)
  add_executable(marian_server command/marian_server.cpp)
  set_target_properties(marian_server PROPERTIES OUTPUT_NAME marian-server)
  target_compile_options(marian_server PUBLIC ${ALL_WARNINGS})
  set(EXECUTABLES ${EXECUTABLES} marian_server)
endif(COMPILE_SERVER)

foreach(exec ${EXECUTABLES})
  target_link_libraries(${exec} marian ${EXT_LIBS} ${EXT_LIBS} ${CMAKE_THREAD_LIBS_INIT})
  if(CUDA_FOUND)
    target_link_libraries(${exec} marian marian_cuda ${EXT_LIBS} ${CMAKE_THREAD_LIBS_INIT})
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
