
include_directories(.)

add_subdirectory(yaml-cpp)

add_library(libcommon OBJECT
  cnpy/cnpy.cpp
  common/exception.cpp
  common/logging.cpp
  common/vocab.cpp
  common/utils.cpp
  gpu/npz_converter.cu
)

#add_library(librescorer OBJECT
#  rescorer/nbest.cpp
#)

add_library(libamun OBJECT
  common/config.cpp
#  decoder/kenlm.cpp
)

cuda_add_executable(
  amun
  gpu/decoder/ape_penalty.cu
  common/decoder_main.cpp
  gpu/decoder/encoder_decoder.cu
  common/god.cpp
  common/history.cpp
  #decoder/language_model.cu
  common/loader.cpp
  common/loader_factory.cu
  common/printer.cpp
  common/scorer.cpp
  common/search.cpp
  common/sentence.cpp

  gpu/mblas/matrix.cu
  gpu/dl4mt/gru.cu

  cpu/mblas/matrix.cpp
  cpu/mblas/phoenix_functions.cpp
  cpu/dl4mt/encoder.cpp
  cpu/dl4mt/model.cpp
  #cpu/decoder/encoder_decoder.cpp
  
  $<TARGET_OBJECTS:libamun>
  $<TARGET_OBJECTS:libcommon>
  $<TARGET_OBJECTS:libyaml-cpp>
)

#cuda_add_executable(
#  rescorer
#  rescorer/rescorer_main.cu
#  mblas/matrix.cu
#  dl4mt/gru.cu
#  $<TARGET_OBJECTS:librescorer>
#  $<TARGET_OBJECTS:libcommon>
#  $<TARGET_OBJECTS:libyaml-cpp>
#)

foreach(exec amun)
  target_link_libraries(${exec} ${EXT_LIBS} cuda)
  cuda_add_cublas_to_target(${exec})
  set_target_properties(${exec} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endforeach(exec)

add_subdirectory(bpe)
