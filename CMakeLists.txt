cmake_minimum_required(VERSION 3.15)
project(monte_carlo VERSION 1.0.0 LANGUAGES CXX)

# Require C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Create library target
add_library(monte_carlo
    src/random.cpp
)

# Set include directories
target_include_directories(monte_carlo
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Enable testing
enable_testing()

# Add the tests subdirectory
add_subdirectory(tests)

# Generate compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)