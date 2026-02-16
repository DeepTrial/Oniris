cmake_minimum_required(VERSION 3.14)

# Suppress CMake deprecation warnings from third-party dependencies
# For CMake 3.24+, use extraction timestamp for FetchContent
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
    cmake_policy(SET CMP0135 NEW)
endif()

# For CMake 3.27+, suppress FindPythonInterp deprecation warning from older pybind11
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.27")
    cmake_policy(SET CMP0148 OLD)
endif()

project(Oniris VERSION 0.1.0 LANGUAGES CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(BUILD_TESTS "Build tests" ON)
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(-g -O0)
    else()
        add_compile_options(-O3)
    endif()
elseif(MSVC)
    add_compile_options(/W4 /WX-)
endif()

# Find packages
find_package(Threads REQUIRED)

# Find Python for generating types from ONNX
find_package(Python3 COMPONENTS Interpreter REQUIRED)

# Generate types.hpp and types.cpp from ONNX TensorProto.DataType
set(GENERATED_TYPES_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/core)
set(TYPES_GENERATOR_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_dtypes.py)

add_custom_command(
    OUTPUT ${GENERATED_TYPES_DIR}/types.hpp ${GENERATED_TYPES_DIR}/types.cpp
    COMMAND ${Python3_EXECUTABLE} ${TYPES_GENERATOR_SCRIPT} ${GENERATED_TYPES_DIR}
    DEPENDS ${TYPES_GENERATOR_SCRIPT}
    COMMENT "Generating DataType definitions from ONNX TensorProto..."
)

add_custom_target(generate_types DEPENDS ${GENERATED_TYPES_DIR}/types.hpp ${GENERATED_TYPES_DIR}/types.cpp)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# Core library source files
set(ONIRIS_CORE_SOURCES
    src/core/logger.cpp
    src/core/types.cpp
    src/ir/node.cpp
    src/ir/graph.cpp
    src/ir/model.cpp
    src/passes/shape_inference.cpp
    src/passes/simplifier.cpp
    src/utils/onnx_utils.cpp
)

# Create core library
add_library(oniris_core STATIC ${ONIRIS_CORE_SOURCES})
add_dependencies(oniris_core generate_types)
target_include_directories(oniris_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(oniris_core PUBLIC Threads::Threads)
set_target_properties(oniris_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Python bindings
if(BUILD_PYTHON_BINDINGS)
    # Use FetchContent to get pybind11 if not present
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/pybind11/CMakeLists.txt)
        # Use local copy if available
        add_subdirectory(third_party/pybind11)
    else()
        # Fetch from GitHub (v2.13+ has fixed CMake 3.27+ deprecation warnings)
        include(FetchContent)
        if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
            FetchContent_Declare(
                pybind11
                GIT_REPOSITORY https://github.com/pybind/pybind11.git
                GIT_TAG v2.13.6
                GIT_SHALLOW TRUE
                DOWNLOAD_EXTRACT_TIMESTAMP TRUE
            )
        else()
            FetchContent_Declare(
                pybind11
                GIT_REPOSITORY https://github.com/pybind/pybind11.git
                GIT_TAG v2.13.6
                GIT_SHALLOW TRUE
            )
        endif()
        FetchContent_MakeAvailable(pybind11)
    endif()
    
    pybind11_add_module(_oniris src/python/bindings.cpp)
    target_link_libraries(_oniris PRIVATE oniris_core)
    set_target_properties(_oniris PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/oniris
    )
endif()

# Tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests/unit)
endif()

# Install targets
install(TARGETS oniris_core
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)
install(DIRECTORY src/
    DESTINATION include/oniris
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)
