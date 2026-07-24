cmake_minimum_required(VERSION 3.18)
project(fresub LANGUAGES CXX)

option(ENABLE_CUDA "Build CUDA feasibility kernels" ON)
if(ENABLE_CUDA)
    enable_language(CUDA)
endif()

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA architecture - use single target to avoid nvlink warnings
# You can override this by setting CMAKE_CUDA_ARCHITECTURES environment variable
if(ENABLE_CUDA AND NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Use a single common architecture to reduce nvlink warnings
    # 75 covers most modern GPUs (RTX 20xx, GTX 16xx series)  
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

if(ENABLE_CUDA)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Find required packages
find_package(Threads REQUIRED)

# Add exopt as a subdirectory (now included in lib/)
add_subdirectory(lib)

# The exopt_lib target will provide all necessary includes through its PUBLIC properties
# No need to manually set include paths - they come automatically from target_link_libraries

# Source files - separate library sources from main
set(CPU_LIB_SOURCES
    src/cpu/graph.cpp
    src/cpu/window.cpp
    src/cpu/aig_utils.cpp
    src/cpu/simulation.cpp
    src/cpu/feasibility.cpp
    src/cpu/synthesis.cpp
    src/cpu/insertion.cpp
)

set(CUDA_SOURCES
    src/cuda/resub_kernels.cu
    src/cuda/resub_kernels_all.cu
)

# Create library for CPU components (without main.cpp)
add_library(fresub_cpu STATIC ${CPU_LIB_SOURCES})

# Link libraries - use PUBLIC so our executables get the includes
target_link_libraries(fresub_cpu PUBLIC 
    Threads::Threads
    exopt_lib  # This brings in all exopt dependencies and includes automatically
    mockturtle  # Add mockturtle library for alternative synthesis
)

if(ENABLE_CUDA)
    # Create CUDA library
    add_library(fresub_cuda STATIC ${CUDA_SOURCES})
    set_target_properties(fresub_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS OFF
    )
else()
    add_library(fresub_cuda STATIC src/cpu/cuda_stubs.cpp)
    target_link_libraries(fresub_cuda PUBLIC fresub_cpu)
endif()

# Main executable - clean resubstitution engine (TEMPORARILY ENABLED FOR TESTING)
add_executable(fresub src/cpu/main.cpp)

# Link libraries - includes come automatically from fresub_cpu's dependencies
target_link_libraries(fresub
    fresub_cpu
    fresub_cuda
    Threads::Threads
)

add_executable(fresub_lut src/cpu/main_lut.cpp)
target_link_libraries(fresub_lut
    fresub_cpu
    fresub_cuda
    Threads::Threads
)

# Note: The nvlink warnings about system libraries (librt, libpthread, libdl) 
# are harmless and expected. They occur because nvlink skips CPU-only libraries
# that are incompatible with CUDA device code, which is correct behavior.

# Compiler flags
target_compile_options(fresub_cpu PRIVATE
    -O3
    -march=native
    -Wall
    -Wextra
    $<$<CONFIG:Debug>:-g>
)

target_compile_options(fresub_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        -O3
        $<$<CONFIG:Debug>:-g -G>
    >
)


# Test executables moved to tests/ directory

# Optional: Add tests (WINDOW TEST ONLY)
option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Installation (DISABLED)
# install(TARGETS fresub DESTINATION bin)
# install(DIRECTORY include/ DESTINATION include/fresub)
