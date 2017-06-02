add_executable(iris_example iris/iris.cpp)
cuda_add_executable(mnist_example mnist/mnist_ffnn.cu)

target_link_libraries(iris_example marian)
target_link_libraries(mnist_example marian)

foreach(exec iris_example mnist_example)
  target_link_libraries(${exec} ${EXT_LIBS})
  cuda_add_cublas_to_target(${exec})
  set_target_properties(${exec} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endforeach(exec)
