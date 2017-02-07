add_executable(extract_lex
               extract-lex-main.cpp
               utils.cpp exception.cpp)

target_link_libraries(extract_lex ${EXT_LIBS})
set_target_properties(extract_lex PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
