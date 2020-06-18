# adding a new file require explicittly modifing the CMakeLists.txt

add_definitions(-DFINTEGER=uint64_t)

include_directories("impl")
FILE(GLOB FaissCppSources *.cpp impl/*.cpp utils/*.cpp)
add_library(faiss OBJECT ${FaissCppSources})
