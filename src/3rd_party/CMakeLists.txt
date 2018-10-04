
include_directories(.)

add_subdirectory(yaml-cpp)
add_subdirectory(SQLiteCpp)

include_directories(./SQLiteCpp/include)
include_directories(./CLI)
