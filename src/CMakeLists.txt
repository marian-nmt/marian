
include_directories(.)
include_directories(dl4mt)
include_directories(common)
include_directories(decoder)
include_directories(mblas)

include_directories(yaml-cpp)
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
  decoder/ape_penalty.cu
  common/decoder_main.cpp
  decoder/encoder_decoder.cu
  common/god.cpp
  common/history.cpp
  #decoder/language_model.cu
  decoder/loader.cpp
  decoder/loader_factory.cu
  decoder/printer.cpp
  common/scorer.cpp
  common/search.cpp
  common/sentence.cpp
  gpu/mblas/matrix.cu
  gpu/dl4mt/gru.cu
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
