
include_directories(.)

add_library(libcommon OBJECT
  cnpy/cnpy.cpp
  exception.cpp
)

cuda_add_library(common_cuda 
  expressions.cu 
  test.cu
  tensor.cu	
  tensor_operators.cu
)

target_link_libraries(common_cuda)

cuda_add_executable(
  marian
  tensor_operators.cu
  $<TARGET_OBJECTS:libcommon>
)

target_link_libraries(marian common_cuda)

cuda_add_executable(
  train_mnist
  train_mnist.cu
  $<TARGET_OBJECTS:libcommon>
)

target_link_libraries(train_mnist common_cuda)

cuda_add_executable(
  validate_mnist
  validate_mnist.cu
  $<TARGET_OBJECTS:libcommon>
)

target_link_libraries(validate_mnist common_cuda)

foreach(exec marian train_mnist validate_mnist)
  target_link_libraries(${exec} ${EXT_LIBS} cuda cudnn)
  cuda_add_cublas_to_target(${exec})
  set_target_properties(${exec} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endforeach(exec)
