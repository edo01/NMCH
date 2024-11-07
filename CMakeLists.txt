cmake_minimum_required(VERSION 3.10)

project(NMCH LANGUAGES CXX CUDA)

set(src_dir src)
set(inc_dir include)
set(exe_dir bin)

# The project uses C++11 features
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

find_package(CUDAToolkit)
if(NOT CUDAToolkit_FOUND)
    message(FATAL_ERROR "CUDA not found")
endif()

# Specify the executable and lib output path
set(EXECUTABLE_OUTPUT_PATH ${exe_dir})

# Set compiler flags for different build types
set(CMAKE_CXX_FLAGS_DEBUG "-g -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Enable CUDA language
enable_language(CUDA) #enable NVCC compiler

# Add CUDA macro for conditional compilation
add_definitions(-DUSE_CUDA)

# Set policy to use CUDA_ARCHITECTURES
cmake_policy(SET CMP0104 NEW)

# Set CUDA_ARCHITECTURES to compute capability 6.1
set(CMAKE_CUDA_ARCHITECTURES 61 62 70 72 75 80 86)


# Source files
set(SOURCES
    ${src_dir}/NMCH/methods/NMCH.cu
    ${src_dir}/NMCH/methods/NMCH_fw_euler.cu
    ${src_dir}/NMCH/random/random.cu
#    ${src_dir}/NMCH/utils/cu_utils.cu
    ${src_dir}/NMCH/utils/utils.cpp
    ${src_dir}/NMCH/test/test.cpp
)

# Add executable
add_executable(NMCH ${SOURCES})

# Include directories
include_directories(NMCH ${inc_dir})

# Link CUDA libraries
target_link_libraries(NMCH ${CUDA_LIBRARIES})

# Set C++ standard
set_target_properties(NMCH PROPERTIES CXX_STANDARD 11)
