cmake_minimum_required(VERSION 3.12)
project(CudaMaxCutPlanted LANGUAGES CXX CUDA)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80)

# Find CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})

# Enable CUDA support
enable_language(CUDA)

# Add executable
add_executable(CudaMaxCutPlanted kernel.cu)

# Link CUDA libraries
target_link_libraries(CudaMaxCutPlanted ${CUDA_LIBRARIES})

## Link directories for CUDA, cuBLAS, and cuSPARSE
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)

# Include directories for cuBLAS and cuSPARSE
include_directories(${CUBLAS_INCLUDE_DIRS} ${CUSPARSE_INCLUDE_DIRS})

# Link cuBLAS and cuSPARSE libraries
target_link_libraries(CudaMaxCutPlanted ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcublas.so)
target_link_libraries(CudaMaxCutPlanted ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcusparse.so)
