
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
  mblas/matrix.cpp
  mblas/phoenix_functions.cpp
  dl4mt/gru.cpp
)

add_library(libamun OBJECT
  decoder/decoder_main.cpp
  decoder/god.cpp
  decoder/sentence.cpp
  decoder/config.cpp
)

add_executable(
  amun
  $<TARGET_OBJECTS:libamun>
  $<TARGET_OBJECTS:libcommon>
  $<TARGET_OBJECTS:libyaml-cpp>
)

foreach(exec amun)
  target_link_libraries(${exec} ${EXT_LIBS} pthread tcmalloc_minimal)
  set_target_properties(${exec} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endforeach(exec)

add_subdirectory(bpe)
