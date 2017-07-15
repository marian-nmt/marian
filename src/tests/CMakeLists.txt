# Unit tests
set(TEST_SOURCES
    graph_tests.cpp
    operator_tests.cpp
)

add_executable(run_tests run_tests.cpp ${TEST_SOURCES})
target_link_libraries(run_tests marian ${EXT_LIBS} Catch)
cuda_add_cublas_to_target(run_tests)
set_target_properties(run_tests PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

add_test(NAME GraphTest COMMAND run_tests)

# Testing apps
add_executable(logger_test logger_test.cpp)
#cuda_add_executable(bn_test bn_test.cu)
cuda_add_executable(pooling_test pooling_test.cu)
cuda_add_executable(dropout_test dropout_test.cu)
cuda_add_executable(marian_test marian_test.cu)
cuda_add_executable(tensor_test tensor_test.cu)


foreach(exec
        logger_test
        dropout_test
        pooling_test
        marian_test
        #bn_test
        tensor_test
)
    target_link_libraries(${exec} marian ${EXT_LIBS})
    cuda_add_cublas_to_target(${exec})
    set_target_properties(${exec} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endforeach(exec)

if(CUDNN_FOUND)
    cuda_add_executable(conv_test conv_test.cu)
    # cuda_add_executable(conv_char_test conv_char_test.cu)

    foreach(exec
            conv_test
            # conv_char_test
    )
        target_link_libraries(${exec} marian ${EXT_LIBS})
        cuda_add_cublas_to_target(${exec})
        set_target_properties(${exec} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
    endforeach(exec)
endif(CUDNN_FOUND)
