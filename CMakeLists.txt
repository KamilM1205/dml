cmake_minimum_required(VERSION 3.20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(dml VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

option(DML_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

if(MSVC)
  set(DML_WARNING_FLAGS /W4)
  if(DML_WARNINGS_AS_ERRORS)
    list(APPEND DML_WARNING_FLAGS /WX)
  endif()
else()
  set(DML_WARNING_FLAGS -Wall -Wextra -Wpedantic)
  if(DML_WARNINGS_AS_ERRORS)
    list(APPEND DML_WARNING_FLAGS -Werror)
  endif()
endif()

if(NOT DML_BUILD_TYPE)
  set(dml_build "STATIC")
  message(STATUS "DML_BUILD_TYPE set to ${dml_build} as none was specified.")
  set(DML_BUILD_TYPE ${dml_build} CACHE STRING "Set build type(STATIC or SHARED)")
endif()

message(STATUS "DML build type: ${DML_BUILD_TYPE}")

if(NOT DML_USE_TESTS)
  set(dml_tests ON)
  message(STATUS "DML_USE_TESTS set to ${dml_tests} as none was specified.")
  set(DML_USE_TESTS ${dml_tests} CACHE BOOL "Set build unit tests")
endif()

message(STATUS "DML use tests: ${DML_USE_TESTS}")

# enable_testing() must live in the top-level list file, or CTest registers nothing.
enable_testing()

add_subdirectory(src)

if(DML_USE_TESTS)
  add_subdirectory(tests)
endif()

find_program(DML_CLANG_FORMAT
  NAMES clang-format
        clang-format-21 clang-format-20 clang-format-19 clang-format-18
        clang-format-17 clang-format-16
  HINTS
    /opt/homebrew/opt/llvm/bin
    /usr/local/opt/llvm/bin
    /Library/Developer/CommandLineTools/usr/bin
)

file(GLOB_RECURSE DML_FORMAT_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp"
)

if(DML_CLANG_FORMAT)
  add_custom_target(format
    COMMAND "${DML_CLANG_FORMAT}" --style=file -i ${DML_FORMAT_SOURCES}
    COMMENT "Formatting sources with clang-format (${DML_CLANG_FORMAT})"
    VERBATIM
  )

  add_custom_target(check-format
    COMMAND "${DML_CLANG_FORMAT}" --style=file --dry-run --Werror ${DML_FORMAT_SOURCES}
    COMMENT "Checking formatting with clang-format (${DML_CLANG_FORMAT})"
    VERBATIM
  )
else()
  message(STATUS "clang-format not found - 'format' and 'check-format' targets will fail.")
  message(STATUS "Install it with: brew install llvm")

  add_custom_target(format
    COMMAND ${CMAKE_COMMAND} -E echo
      "clang-format was not found. Install it with: brew install llvm"
    COMMAND ${CMAKE_COMMAND} -E false
    VERBATIM
  )

  add_custom_target(check-format
    COMMAND ${CMAKE_COMMAND} -E echo
      "clang-format was not found. Install it with: brew install llvm"
    COMMAND ${CMAKE_COMMAND} -E false
    VERBATIM
  )
endif()
