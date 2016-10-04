
include_directories(.)
include_directories(3rd_party)

add_subdirectory(3rd_party)



#add_library(librescorer OBJECT
#  rescorer/nbest.cpp
#)


if(CUDA_FOUND)
cuda_add_executable(
  amun
  common/decoder_main.cpp
  common/config.cpp
  common/exception.cpp
  common/loader_factory.cpp
  common/logging.cpp
  common/vocab.cpp
  common/utils.cpp
  common/god.cpp
  common/history.cpp
  common/loader.cpp
  common/printer.cpp
  common/scorer.cpp
  common/search.cpp
  common/sentence.cpp
  common/processor/bpe.cpp
  cpu/mblas/matrix.cpp
  cpu/mblas/phoenix_functions.cpp
  cpu/dl4mt/decoder.cpp
  cpu/dl4mt/encoder.cpp
  cpu/dl4mt/gru.cpp
  cpu/dl4mt/model.cpp
  cpu/decoder/encoder_decoder.cpp
  common/loader_factory.cu
  gpu/decoder/ape_penalty.cu
  gpu/decoder/encoder_decoder.cu
  gpu/mblas/matrix.cu
  gpu/dl4mt/encoder.cu
  gpu/dl4mt/gru.cu
  gpu/npz_converter.cu
  $<TARGET_OBJECTS:libcommon>
  $<TARGET_OBJECTS:libyaml-cpp>
)
else(CUDA_FOUND)
add_executable(
  amun
  common/decoder_main.cpp
  common/config.cpp
  common/exception.cpp
  common/loader_factory.cpp
  common/logging.cpp
  common/vocab.cpp
  common/utils.cpp
  common/god.cpp
  common/history.cpp
  common/loader.cpp
  common/printer.cpp
  common/scorer.cpp
  common/search.cpp
  common/sentence.cpp
  common/processor/bpe.cpp
  cpu/mblas/matrix.cpp
  cpu/mblas/phoenix_functions.cpp
  cpu/dl4mt/decoder.cpp
  cpu/dl4mt/encoder.cpp
  cpu/dl4mt/gru.cpp
  cpu/dl4mt/model.cpp
  cpu/decoder/encoder_decoder.cpp
  $<TARGET_OBJECTS:libcommon>
  $<TARGET_OBJECTS:libyaml-cpp>
)
endif(CUDA_FOUND)


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
  if(CUDA_FOUND)
    target_link_libraries(${exec} ${EXT_LIBS} cuda)
    cuda_add_cublas_to_target(${exec})
  else(CUDA_FOUND)
    target_link_libraries(${exec} ${EXT_LIBS})
  endif(CUDA_FOUND)
  set_target_properties(${exec} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endforeach(exec)

add_subdirectory(bpe)
