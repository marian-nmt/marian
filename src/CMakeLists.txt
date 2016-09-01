
include_directories(.)
include_directories(dl4mt)
include_directories(common)
include_directories(decoder)
include_directories(mblas)

include_directories(yaml-cpp)
add_subdirectory(yaml-cpp)

add_library(libcommon OBJECT
  cnpy/cnpy.cpp
  common/utils.cpp
  common/exception.cpp
)

add_library(librescorer OBJECT
  rescorer/nbest.cpp
)

add_library(libamun OBJECT
  decoder/config.cpp
#  decoder/kenlm.cpp
)

cuda_add_executable(
  amun
  decoder/decoder_main.cu
  decoder/god.cu
  decoder/search.cu
  decoder/sentence.cu
  mblas/matrix.cu
  dl4mt/gru.cu
  $<TARGET_OBJECTS:libamun>
  $<TARGET_OBJECTS:libcommon>
  $<TARGET_OBJECTS:libyaml-cpp>
)

cuda_add_executable(
  rescorer
  rescorer/rescorer_main.cu
  mblas/matrix.cu
  dl4mt/gru.cu
  $<TARGET_OBJECTS:librescorer>
  $<TARGET_OBJECTS:libcommon>
  $<TARGET_OBJECTS:libyaml-cpp>
)

foreach(exec amun rescorer)
  target_link_libraries(${exec} ${EXT_LIBS} cuda)
  cuda_add_cublas_to_target(${exec})
  set_target_properties(${exec} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endforeach(exec)

add_subdirectory(bpe)
