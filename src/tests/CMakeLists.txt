# Unit tests
set(UNIT_TESTS
    graph_tests
    operator_tests
    rnn_tests
    attention_tests
)

foreach(test ${UNIT_TESTS})
    add_executable("run_${test}" run_tests.cpp "${test}.cpp")
    target_link_libraries("run_${test}" marian ${EXT_LIBS} Catch)
    cuda_add_cublas_to_target("run_${test}")

    add_test(NAME ${test} COMMAND "run_${test}")
endforeach(test)


# Testing apps
add_executable(logger_test logger_test.cpp)
#cuda_add_executable(bn_test bn_test.cu)
cuda_add_executable(pooling_test pooling_test.cu)
cuda_add_executable(dropout_test dropout_test.cu)
#cuda_add_executable(marian_test marian_test.cu)
cuda_add_executable(tensor_test tensor_test.cu)

add_executable(sqlite_test sqlite_test.cpp)

foreach(exec
        logger_test
        dropout_test
        pooling_test
        #marian_test
        #bn_test
        tensor_test
        sqlite_test
)
    target_link_libraries(${exec} marian ${EXT_LIBS})
    cuda_add_cublas_to_target(${exec})
    set_target_properties(${exec} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endforeach(exec)

# if(CUDNN_FOUND)
    # cuda_add_executable(conv_test conv_test.cu)

    # foreach(exec
            # conv_test
    # )
        # target_link_libraries(${exec} marian ${EXT_LIBS})
        # cuda_add_cublas_to_target(${exec})
        # set_target_properties(${exec} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
    # endforeach(exec)
# endif(CUDNN_FOUND)
