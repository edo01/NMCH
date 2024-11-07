cmake_minimum_required(VERSION 3.10)

project(NMCH LANGUAGES CXX CUDA)

# Check for CUDA
find_package(CUDA REQUIRED)

set(src_dir src)
set(inc_dir include)
set(exe_dir bin)

# The project uses C++11 features
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Specify the executable and lib output path
set(EXECUTABLE_OUTPUT_PATH ${exe_dir})

# Include directories
include_directories(${inc_dir})

# Source files
set(SOURCES
    ${src_dir}/NMCH/methods/NMCH.cpp
    ${src_dir}/NMCH/methods/NMCH_fw_euler.cu
    ${src_dir}/NMCH/random/random.cu
    ${src_dir}/NMCH/utils/cu_utils.cu
    ${src_dir}/NMCH/utils/utils.cpp
    ${src_dir}/NMCH/test/test.cpp
)

# Add executable
add_executable(NMCH ${SOURCES})

# Link CUDA libraries
target_link_libraries(NMCH ${CUDA_LIBRARIES})

# Set C++ standard
set_target_properties(NMCH PROPERTIES CXX_STANDARD 11)