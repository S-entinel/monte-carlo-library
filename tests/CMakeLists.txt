include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.12.1
)
FetchContent_MakeAvailable(googletest)

# Add test executable
add_executable(monte_carlo_tests 
    tests.cpp
)

target_link_libraries(monte_carlo_tests
    PRIVATE
        monte_carlo
        GTest::gtest_main
)

# Add test
include(GoogleTest)
gtest_discover_tests(monte_carlo_tests)