
include_directories(.)

add_library(libcommon OBJECT
  cnpy/cnpy.cpp
  exception.cpp
)

cuda_add_executable(
  marian
  expressions.cu
  test.cu
  tensor.cu
  tensor_operators.cu
  $<TARGET_OBJECTS:libcommon>
)

cuda_add_executable(
  train_mnist
  expressions.cu
  train_mnist.cu
  tensor.cu
  tensor_operators.cu
  $<TARGET_OBJECTS:libcommon>
)

cuda_add_executable(
  validate_mnist
  expressions.cu
  validate_mnist.cu
  tensor.cu
  tensor_operators.cu
  $<TARGET_OBJECTS:libcommon>
)

foreach(exec marian train_mnist validate_mnist)
  target_link_libraries(${exec} ${EXT_LIBS} cuda cudnn)
  cuda_add_cublas_to_target(${exec})
  set_target_properties(${exec} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endforeach(exec)
