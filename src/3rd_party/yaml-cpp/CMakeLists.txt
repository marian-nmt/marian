include_directories(..)
include_directories(.)

FILE(GLOB YamlCppSources *.cpp contrib/*.cpp)
if (NOT TARGET libyaml-cpp)
  add_library(libyaml-cpp OBJECT ${YamlCppSources})
endif()
