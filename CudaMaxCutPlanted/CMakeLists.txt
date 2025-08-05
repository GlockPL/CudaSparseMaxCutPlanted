# Find required packages
find_package(CUDAToolkit REQUIRED)

# Add executable with all source files
add_executable(CudaMaxCutPlanted 
    main.cu 
    kernel.cu 
    Kernels.cu 
    CudaSparseMatrix.cu
)

# Set properties for CUDA compilation
set_property(TARGET CudaMaxCutPlanted PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})

# Include directories
target_include_directories(CudaMaxCutPlanted PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Link CUDA libraries using modern CMake targets
target_link_libraries(CudaMaxCutPlanted 
    PRIVATE 
    CUDA::cudart
    CUDA::cublas
    CUDA::cusparse
    CUDA::curand
)

# Set compiler flags for better compatibility
if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    target_compile_options(CudaMaxCutPlanted PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            --extended-lambda
            --expt-relaxed-constexpr
        >
    )
endif()

# Set runtime library path on Linux
if(UNIX AND NOT APPLE)
    set_target_properties(CudaMaxCutPlanted PROPERTIES
        BUILD_RPATH_USE_ORIGIN TRUE
        INSTALL_RPATH "$ORIGIN/../lib:$ORIGIN/"
    )
endif()
