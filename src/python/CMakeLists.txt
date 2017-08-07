add_library(python SHARED
  mariannmt.cpp
)

set_target_properties(python PROPERTIES EXCLUDE_FROM_ALL 1)
set_target_properties(python PROPERTIES OUTPUT_NAME mariannmt)
set_target_properties(python PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set_target_properties(python PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

target_link_libraries(python ${EXT_LIBS})
