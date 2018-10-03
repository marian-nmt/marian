
include_directories(.)

add_subdirectory(yaml-cpp)
add_subdirectory(SQLiteCpp)

set(SPM_ENABLE_SHARED OFF CACHE BOOL "Builds shared libaries in addition to static libraries." FORCE)
set(SPM_ENABLE_TCMALLOC ON CACHE BOOL "Enable TCMalloc if available." FORCE)
set(SPM_TCMALLOC_STATIC ON CACHE BOOL "Link static library of TCMALLOC." FORCE)

add_subdirectory(sentencepiece)

include_directories(./SQLiteCpp/include)
include_directories(./CLI)
include_directories(./sentencepiece)
