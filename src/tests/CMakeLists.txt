# Unit tests
add_subdirectory(units)

if(NOT MSVC)
  # Testing apps
  set(APP_TESTS
      logger
      dropout
      sqlite
      prod
      cli
      pooling
  )

  foreach(test ${APP_TESTS})
    add_executable("test_${test}" "${test}.cpp")

    if(CUDA_FOUND)
      target_link_libraries("test_${test}" ${EXT_LIBS} marian ${EXT_LIBS} marian_cuda ${EXT_LIBS})
    else(CUDA_FOUND)
      target_link_libraries("test_${test}" marian ${EXT_LIBS})
    endif(CUDA_FOUND)

    set_target_properties("test_${test}" PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
  endforeach(test)
endif(NOT MSVC)