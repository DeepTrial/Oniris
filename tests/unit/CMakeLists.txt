# Unit tests for Oniris

include(FetchContent)

# Use extraction timestamp for downloaded files (CMake 3.24+)
# Using v1.14.0 which has fixed CMake 3.27+ deprecation warnings
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
else()
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
    )
endif()
FetchContent_MakeAvailable(googletest)

enable_testing()

set(TEST_SOURCES
    test_types.cpp
    test_ir.cpp
    test_shape_inference.cpp
    test_simplifier.cpp
)

add_executable(oniris_tests ${TEST_SOURCES})

target_link_libraries(oniris_tests
    oniris_core
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(oniris_tests)
