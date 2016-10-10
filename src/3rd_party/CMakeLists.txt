
include_directories(.)

add_subdirectory(yaml-cpp)
add_subdirectory(fast_align)
add_subdirectory(extract_lex)

add_library(libcnpy OBJECT
  cnpy/cnpy.cpp
)
