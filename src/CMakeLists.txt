
add_subdirectory(amun)

if(CUDA_FOUND)
  if(MARIAN)
    add_subdirectory(marian)
  endif(MARIAN)
endif(CUDA_FOUND)


