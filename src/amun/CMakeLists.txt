
include_directories(.)
include_directories(3rd_party)

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/common/git_version.cpp.in"
               "${CMAKE_CURRENT_BINARY_DIR}/common/git_version.cpp" @ONLY)

list(APPEND SOURCES "${CMAKE_CURRENT_BINARY_DIR}/common/git_version.cpp")


add_library(cpumode OBJECT
  cpu/mblas/matrix.cpp
  cpu/mblas/phoenix_functions.cpp
  cpu/decoder/encoder_decoder.cpp
  cpu/decoder/encoder_decoder_state.cpp
  cpu/decoder/encoder_decoder_loader.cpp
  cpu/dl4mt/encoder.cpp
  cpu/dl4mt/gru.cpp
  cpu/dl4mt/model.cpp
  cpu/dl4mt/encoder_decoder.cpp
  cpu/nematus/encoder.cpp
  cpu/nematus/model.cpp
  cpu/nematus/gru.cpp
  cpu/nematus/transition.cpp
  cpu/nematus/encoder_decoder.cpp

  #fpga/best_hyps.cpp
  #fpga/decoder.cpp
  #fpga/encoder_decoder_loader.cpp
  #fpga/encoder_decoder_state.cpp
  #fpga/encoder_decoder.cpp
  #fpga/encoder.cpp
  #fpga/matrix_functions.cpp
  #fpga/matrix.cpp
  #fpga/model.cpp
  #fpga/nth_element.cpp
  #fpga/npz_converter.cpp
  #fpga/kernel.cpp
  #fpga/debug-devices.cpp
  #fpga/hello_world.cpp
)

add_library(libcommon OBJECT
  ${CMAKE_CURRENT_BINARY_DIR}/common/git_version.cpp
  common/base_matrix.cpp
  common/config.cpp
  common/exception.cpp
  common/filter.cpp
  common/god.cpp
  common/history.cpp
  common/hypothesis.cpp
  common/loader.cpp
  common/logging.cpp
  common/output_collector.cpp
  common/printer.cpp
  common/processor/bpe.cpp
  common/scorer.cpp
  common/search.cpp
  common/sentence.cpp
  common/sentences.cpp
  common/types.cpp
  common/utils.cpp
  common/vocab.cpp
  common/translation_task.cpp
)

if(CUDA_FOUND)

if (MARIAN)
   #execute_process(COMMAND git submodule update --init --recursive
#                  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
#   add_subdirectory(marian)
endif(MARIAN)  

set_source_files_properties( common/loader_factory.cpp
  PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ )

cuda_add_executable(
  amun
  common/decoder_main.cpp
  gpu/decoder/encoder_decoder.cu
  gpu/decoder/encoder_decoder_loader.cu
  gpu/decoder/encoder_decoder_state.cu
  gpu/dl4mt/encoder.cu
  gpu/dl4mt/gru.cu
  gpu/mblas/matrix.cu
  gpu/mblas/matrix_functions.cu
  gpu/mblas/nth_element.cu
  gpu/npz_converter.cu
  gpu/types-gpu.cu


  common/loader_factory.cpp
  $<TARGET_OBJECTS:libcommon>
  $<TARGET_OBJECTS:cpumode>
  $<TARGET_OBJECTS:libyaml-cpp-amun>
  $<TARGET_OBJECTS:libcnpy>
)

if(PYTHONLIBS_FOUND)
cuda_add_library(python SHARED
  python/amunmt.cpp
  # gpu/decoder/ape_penalty.cu
  gpu/decoder/encoder_decoder.cu
  gpu/decoder/encoder_decoder_loader.cu
  gpu/decoder/encoder_decoder_state.cu
  gpu/mblas/matrix.cu
  gpu/mblas/matrix_functions.cu
  gpu/mblas/nth_element.cu
  gpu/dl4mt/encoder.cu
  gpu/dl4mt/gru.cu
  gpu/npz_converter.cu
  gpu/types-gpu.cu
  common/loader_factory.cpp
  $<TARGET_OBJECTS:libcommon>
  $<TARGET_OBJECTS:libcnpy>
  $<TARGET_OBJECTS:cpumode>
  $<TARGET_OBJECTS:libyaml-cpp-amun>
)
set_target_properties("python" PROPERTIES EXCLUDE_FROM_ALL 1)
set_target_properties("python" PROPERTIES OUTPUT_NAME "amunmt")

endif(PYTHONLIBS_FOUND)

cuda_add_library(mosesplugin STATIC
  plugin/hypo_info.cpp
  #plugin/nbest.cu
  plugin/nmt.cu
  plugin/neural_phrase.cpp
  gpu/decoder/encoder_decoder.cu
  gpu/decoder/encoder_decoder_loader.cu
  gpu/decoder/encoder_decoder_state.cu
  gpu/mblas/matrix.cu
  gpu/mblas/matrix_functions.cu
  gpu/mblas/nth_element.cu
  gpu/dl4mt/encoder.cu
  gpu/dl4mt/gru.cu
  gpu/npz_converter.cu
  gpu/types-gpu.cu
  common/loader_factory.cpp
  $<TARGET_OBJECTS:libcommon>
  $<TARGET_OBJECTS:libcnpy>
  $<TARGET_OBJECTS:cpumode>
  $<TARGET_OBJECTS:libyaml-cpp-amun>
)
set_target_properties("mosesplugin" PROPERTIES EXCLUDE_FROM_ALL 1)

else(CUDA_FOUND)

add_executable(
  amun
  common/decoder_main.cpp
  common/loader_factory.cpp
  $<TARGET_OBJECTS:libcnpy>
  $<TARGET_OBJECTS:cpumode>
  $<TARGET_OBJECTS:libcommon>
  $<TARGET_OBJECTS:libyaml-cpp-amun>
)

if(PYTHONLIBS_FOUND)
add_library(python SHARED
  python/amunmt.cpp
  common/loader_factory.cpp
  $<TARGET_OBJECTS:libcnpy>
  $<TARGET_OBJECTS:cpumode>
  $<TARGET_OBJECTS:libcommon>
  $<TARGET_OBJECTS:libyaml-cpp-amun>
)
set_target_properties("python" PROPERTIES EXCLUDE_FROM_ALL 1)
set_target_properties("python" PROPERTIES OUTPUT_NAME "amunmt")
endif(PYTHONLIBS_FOUND)
endif(CUDA_FOUND)

SET(EXES "amun")

if(PYTHONLIBS_FOUND)
SET(EXES ${EXES} "python")
endif(PYTHONLIBS_FOUND)

foreach(exec ${EXES})
  if(CUDA_FOUND)
    target_link_libraries(${exec} ${EXT_LIBS})
    cuda_add_cublas_to_target(${exec})
  else(CUDA_FOUND)
    target_link_libraries(${exec} ${EXT_LIBS})
  endif(CUDA_FOUND)
  set_target_properties(${exec} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endforeach(exec)

add_subdirectory(3rd_party)
