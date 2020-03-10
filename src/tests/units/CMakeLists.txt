# Unit tests
set(UNIT_TESTS
    graph_tests
    operator_tests
    rnn_tests
    attention_tests
    fastopt_tests
)

foreach(test ${UNIT_TESTS})
  add_executable("run_${test}" run_tests.cpp "${test}.cpp")

  if(CUDA_FOUND)
    target_link_libraries("run_${test}" ${EXT_LIBS} marian ${EXT_LIBS} marian_cuda ${EXT_LIBS} Catch)
  else(CUDA_FOUND)
    target_link_libraries("run_${test}" marian ${EXT_LIBS} Catch)
  endif(CUDA_FOUND)

  add_test(NAME ${test} COMMAND "run_${test}")
endforeach(test)
