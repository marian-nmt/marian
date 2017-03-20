include_directories(..)
include_directories(.)

FILE(GLOB YamlCppSources *.cpp contrib/*.cpp)
add_library(libyaml-cpp OBJECT ${YamlCppSources})