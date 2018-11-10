include_directories(..)
include_directories(.)
include_directories(include)

FILE(GLOB PathieCppSources src/*.cpp)
if (NOT TARGET pathie-cpp)
  add_library(pathie-cpp OBJECT ${PathieCppSources})
endif()
