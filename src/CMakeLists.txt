
include_directories(.)

add_library(libcommon OBJECT
  exception.cpp
)

cuda_add_executable(
  marian
  test.cu
  expressions.cu
  tensor_operators.cu
  $<TARGET_OBJECTS:libcommon>
)

foreach(exec marian)
  target_link_libraries(${exec} ${EXT_LIBS} cuda cudnn)
  cuda_add_cublas_to_target(${exec})
  set_target_properties(${exec} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endforeach(exec)
