# adding a new file require explicittly modifing the CMakeLists.txt

include_directories("impl")

FILE(GLOB FaissCppSources *.cpp impl/*.cpp utils/*.cpp)

if (NOT TARGET faiss)
  add_library(faiss STATIC ${FaissCppSources})
endif()

