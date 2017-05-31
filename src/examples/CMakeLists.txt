cuda_add_executable(iris_example iris/iris.cu)
cuda_add_executable(mnist_example mnist/mnist_ffnn.cu)
cuda_add_executable(mnist_mgpus mnist/mnist_mgpus.cu)

target_link_libraries(iris_example marian_lib)
target_link_libraries(mnist_example marian_lib)
target_link_libraries(mnist_mgpus marian_lib)

foreach(exec iris_example mnist_example mnist_mgpus)
  target_link_libraries(${exec} ${EXT_LIBS})
  cuda_add_cublas_to_target(${exec})
  set_target_properties(${exec} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endforeach(exec)
