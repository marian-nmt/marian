
include_directories(.)

cuda_add_library(marian_lib 
  cnpy/cnpy.cpp
  exception.cpp
  expressions.cu 
  test.cu
  tensor.cu	
  tensor_operators.cu
)

target_link_libraries(marian_lib)

cuda_add_executable(
  marian
  tensor_operators.cu
)

target_link_libraries(marian marian_lib)

cuda_add_executable(
  train_mnist
  train_mnist.cu
)

target_link_libraries(train_mnist marian_lib)

cuda_add_executable(
  validate_mnist
  validate_mnist.cu
)

target_link_libraries(validate_mnist marian_lib)

foreach(exec marian train_mnist validate_mnist)
  target_link_libraries(${exec} ${EXT_LIBS} cuda cudnn)
  cuda_add_cublas_to_target(${exec})
  set_target_properties(${exec} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endforeach(exec)
