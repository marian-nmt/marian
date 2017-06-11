add_executable(logger_test logger_test.cpp)

cuda_add_executable(dropout_test dropout_test.cu)
cuda_add_executable(allocator_test allocator_test.cu)
#cuda_add_executable(tensor_test tensor_test.cu)
#cuda_add_executable(marian_test marian_test.cu)
cuda_add_executable(bn_test bn_test.cu)

foreach(exec
    logger_test
    dropout_test
    allocator_test
    #tensor_test
    #marian_test
    bn_test)
  target_link_libraries(${exec} marian ${EXT_LIBS})
  cuda_add_cublas_to_target(${exec})
  set_target_properties(${exec} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endforeach(exec)
