include(FetchContent)

set(FETCHCONTENT_QUIET OFF)

FetchContent_Declare(googletest
  GIT_REPOSITORY https://github.com/google/googletest
  GIT_TAG v1.18.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

set(BASE_LIBS
  GTest::gtest_main
  dml::dml
)

add_executable(
  test_vector
  ${CMAKE_CURRENT_LIST_DIR}/test_vector.cpp
)
target_link_libraries(
  test_vector
  ${BASE_LIBS}
)
target_compile_options(test_vector PRIVATE ${DML_WARNING_FLAGS})

include(GoogleTest)
gtest_discover_tests(test_vector)
