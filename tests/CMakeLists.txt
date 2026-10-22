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

# One executable per test suite, named test_<suite>, from tests/test_<suite>.cpp.
# Adding a unit means adding one line here and one file in tests/.
set(DML_TEST_SUITES
  dml
  vector
)

include(GoogleTest)

foreach(suite IN LISTS DML_TEST_SUITES)
  add_executable(test_${suite} ${CMAKE_CURRENT_LIST_DIR}/test_${suite}.cpp)
  target_link_libraries(test_${suite} PRIVATE ${BASE_LIBS})
  target_compile_options(test_${suite} PRIVATE ${DML_WARNING_FLAGS})
  gtest_discover_tests(test_${suite})
endforeach()
