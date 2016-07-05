
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
  common/filter.cpp
  mblas/matrix.cpp
  mblas/phoenix_functions.cpp
  dl4mt/encoder.cpp
  dl4mt/gru.cpp
  dl4mt/model.cpp
)

add_library(libamun OBJECT
  decoder/god.cpp
  decoder/search.cpp
  decoder/sentence.cpp
  decoder/config.cpp
  decoder/history.cpp
  decoder/hypothesis.cpp
)

add_library(amunmt SHARED
  python/amunmt.cpp
  $<TARGET_OBJECTS:libamun>
  $<TARGET_OBJECTS:libcommon>
  $<TARGET_OBJECTS:libyaml-cpp>
)

add_executable(
  amun
  decoder/decoder_main.cpp
  $<TARGET_OBJECTS:libamun>
  $<TARGET_OBJECTS:libcommon>
  $<TARGET_OBJECTS:libyaml-cpp>
)

add_executable(
  experimental
  test/test_main.cpp
  $<TARGET_OBJECTS:libamun>
  $<TARGET_OBJECTS:libcommon>
  $<TARGET_OBJECTS:libyaml-cpp>
)

foreach(exec amun amunmt experimental)
  target_link_libraries(${exec} ${EXT_LIBS} tcmalloc_minimal)
  set_target_properties(${exec} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endforeach(exec)

add_subdirectory(bpe)
