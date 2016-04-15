add_executable(bpe bpe_main.cpp)
target_link_libraries(bpe ${EXT_LIBS})

set_target_properties(bpe
	PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
