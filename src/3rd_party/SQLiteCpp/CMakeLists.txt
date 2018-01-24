# Main CMake file for compiling the library itself, examples and tests.
#
# Copyright (c) 2012-2016 Sebastien Rombauts (sebastien.rombauts@gmail.com)
#
# Distributed under the MIT License (MIT) (See accompanying file LICENSE.txt
# or copy at http://opensource.org/licenses/MIT)

# Options relative to SQLite and SQLiteC++ functions

option(SQLITE_ENABLE_COLUMN_METADATA "Enable Column::getColumnOriginName(). Require support from sqlite3 library." ON)
if (SQLITE_ENABLE_COLUMN_METADATA)
    # Enable the use of SQLite column metadata and Column::getColumnOriginName() method,
    # Require that the sqlite3 library is also compiled with this flag (default under Debian/Ubuntu, but not on Mac OS X).
    add_definitions(-DSQLITE_ENABLE_COLUMN_METADATA)
endif (SQLITE_ENABLE_COLUMN_METADATA)

option(SQLITE_ENABLE_ASSERT_HANDLER "Enable the user defintion of a assertion_failed() handler." OFF)
if (SQLITE_ENABLE_ASSERT_HANDLER)
    # Enable the user defintion of a assertion_failed() handler (default to false, easier to handler for begginers).
    add_definitions(-DSQLITECPP_ENABLE_ASSERT_HANDLER)
endif (SQLITE_ENABLE_ASSERT_HANDLER)

option(SQLITE_USE_LEGACY_STRUCT "Fallback to forward declaration of legacy struct sqlite3_value (pre SQLite 3.19)" OFF)
if (SQLITE_USE_LEGACY_STRUCT)
    # Force forward declaration of legacy struct sqlite3_value (pre SQLite 3.19)
    add_definitions(-DSQLITE_USE_LEGACY_STRUCT)
endif (SQLITE_USE_LEGACY_STRUCT)


## Build the C++ Wrapper ##

# adding a new file require explicittly modifing the CMakeLists.txt
# so that CMake knows that it should rebuild the project (it is best practice)

# list of sources files of the library
set(SQLITECPP_SRC
 src/Backup.cpp
 src/Column.cpp
 src/Database.cpp
 src/Exception.cpp
 src/Statement.cpp
 src/Transaction.cpp
 sqlite3/sqlite3.c
)
source_group(src FILES ${SQLITECPP_SRC})

# list of header files of the library
set(SQLITECPP_INC
 include/SQLiteCpp/SQLiteCpp.h
 include/SQLiteCpp/Assertion.h
 include/SQLiteCpp/Backup.h
 include/SQLiteCpp/Column.h
 include/SQLiteCpp/Database.h
 include/SQLiteCpp/Exception.h
 include/SQLiteCpp/Statement.h
 include/SQLiteCpp/Transaction.h
 include/SQLiteCpp/Utils.h
 include/SQLiteCpp/VariadicBind.h
)
source_group(include FILES ${SQLITECPP_INC})

# All includes are relative to the "include" directory
include_directories("include")
include_directories("sqlite3")

# add sources of the wrapper as a "SQLiteCpp" static library
add_library(SQLiteCpp OBJECT ${SQLITECPP_SRC} ${SQLITECPP_INC} sqlite3)

if (UNIX AND (CMAKE_COMPILER_IS_GNUCXX OR ${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang"))
    set_target_properties(SQLiteCpp PROPERTIES COMPILE_FLAGS "-fPIC")
endif (UNIX AND (CMAKE_COMPILER_IS_GNUCXX OR ${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang"))


