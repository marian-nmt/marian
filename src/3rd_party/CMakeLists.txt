
include_directories(.)

add_subdirectory(yaml-cpp)

add_library(libcommon OBJECT
  cnpy/cnpy.cpp
)
