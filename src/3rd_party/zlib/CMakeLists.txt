# list of sources files of the library
file(GLOB ZLIB_SRC *.c)
file(GLOB ZLIB_INC *.h)

# add sources of the wrapper as a "SQLiteCpp" static library
add_library(zlib OBJECT ${ZLIB_SRC} ${ZLIB_INC})
target_compile_options(zlib PUBLIC -Wno-implicit-function-declaration)
