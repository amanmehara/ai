include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.10.0
)
FetchContent_MakeAvailable(googletest)

enable_testing()
include(GoogleTest)

add_executable(ai_test)

set_target_properties(ai_test PROPERTIES LINKER_LANGUAGE CXX)

enable_testing()
add_subdirectory(ai)

target_link_libraries(ai_test ai)
target_link_libraries(ai_test gtest gtest_main)
gtest_discover_tests(ai_test)