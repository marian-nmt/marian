include_directories(..)
include_directories(.)

FILE(GLOB QuicksandSources *.cpp)
add_library(marian4quicksand STATIC ${QuicksandSources})

message(${EXT_LIBS})
target_link_libraries(marian4quicksand marian ${EXT_LIBS})

set_target_properties(marian4quicksand PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set_target_properties(marian4quicksand PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
