cmake_minimum_required(VERSION 3.18)
project(CudaMaxCutPlanted LANGUAGES CXX CUDA)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA toolkit
find_package(CUDAToolkit REQUIRED)

# Set CUDA architectures for GitHub Actions compatibility
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "60;70;75;80;86")
endif()

# Add subdirectory where the source files are located
add_subdirectory(CudaMaxCutPlanted)
