# list of sources files of the library
file(GLOB ZLIB_SRC *.c)
file(GLOB ZLIB_INC *.h)

# add sources of the wrapper as a "zlib" static library
add_library(zlib OBJECT ${ZLIB_SRC} ${ZLIB_INC})

if(MSVC)
  target_compile_options(zlib PUBLIC /wd4996 /wd4267)
else()
  target_compile_options(zlib PUBLIC -Wno-implicit-function-declaration)
endif()
