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

    if(CUDA_FOUND)
      target_link_libraries("run_${test}" marian marian_cuda ${EXT_LIBS} Catch)
      cuda_add_cublas_to_target("run_${test}")
    endif(CUDA_FOUND)

    add_test(NAME ${test} COMMAND "run_${test}")
endforeach(test)


# Testing apps
add_executable(logger_test logger_test.cpp)
add_executable(dropout_test dropout_test.cpp)

if(CUDA_FOUND)
cuda_add_executable(pooling_test pooling_test.cu)
target_link_libraries(pooling_test marian marian_cuda ${EXT_LIBS} Catch)
cuda_add_cublas_to_target(pooling_test)
endif(CUDA_FOUND)

add_executable(sqlite_test sqlite_test.cpp)
foreach(exec
        logger_test
        dropout_test
        sqlite_test
    )
    target_link_libraries(${exec} marian ${EXT_LIBS} Catch)
    if(CUDA_FOUND)
      target_link_libraries(${exec} marian_cuda)
      cuda_add_cublas_to_target(${exec})
    endif(CUDA_FOUND)

    set_target_properties(${exec} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endforeach(exec)

