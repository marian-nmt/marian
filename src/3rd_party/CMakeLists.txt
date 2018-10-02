
include_directories(.)

add_subdirectory(yaml-cpp)
add_subdirectory(SQLiteCpp)
add_subdirectory(sentencepiece)

include_directories(./SQLiteCpp/include)
include_directories(./CLI)
include_directories(./sentencepiece)
