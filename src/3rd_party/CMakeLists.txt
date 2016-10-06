
include_directories(.)

add_subdirectory(yaml-cpp)

add_library(libcnpy OBJECT
  cnpy/cnpy.cpp
)
