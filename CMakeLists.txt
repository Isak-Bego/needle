cmake_minimum_required(VERSION 3.29)
project(needle LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# -------- My Header-only library --------
add_library(needle_lib INTERFACE)
target_include_directories(needle_lib
        INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# -------- Original demo app --------
add_executable(needle)
target_sources(needle PRIVATE main.cpp)
target_link_libraries(needle PRIVATE needle_lib)


# -------- GoogleTest --------
include(FetchContent)
FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.12.1
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(tests
        tests/unit/test_autoGradEngine.cpp
        tests/unit/test_nnComponents.cpp
        tests/integration/test_models.cpp
        tests/unit/test_lossFunctions.cpp
        tests/unit/test_optimizerUtils.cpp
        tests/unit/test_robustness.cpp
)
target_link_libraries(tests
        PRIVATE
        needle_lib
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(tests)