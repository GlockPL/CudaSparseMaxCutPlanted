#include <iostream>
#include <cuda_runtime.h>
#include "CudaSparseMatrix.hpp" // For CHECK_CUDA

void printGpuProperties() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "No CUDA-enabled devices found." << std::endl;
        return;
    }

    std::cout << "Found " << deviceCount << " CUDA-enabled device(s)." << std::endl;

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp devProp;
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaGetDeviceProperties(&devProp, i));

        std::cout << "\n--- Device " << i << ": " << devProp.name << " ---" << std::endl;
        std::cout << "  CUDA Capability Major/Minor version: " << devProp.major << "." << devProp.minor << std::endl;
        std::cout << "  Total global memory: " << devProp.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Shared memory per block: " << devProp.sharedMemPerBlock / 1024 << " KB" << std::endl;
        std::cout << "  Registers per block: " << devProp.regsPerBlock << std::endl;
        std::cout << "  Warp size: " << devProp.warpSize << " threads" << std::endl;
        std::cout << "  Max threads per block: " << devProp.maxThreadsPerBlock << std::endl;
        std::cout << "  Max threads dimensions: (" << devProp.maxThreadsDim[0] << ", " << devProp.maxThreadsDim[1] << ", " << devProp.maxThreadsDim[2] << ")" << std::endl;
        std::cout << "  Max grid size: (" << devProp.maxGridSize[0] << ", " << devProp.maxGridSize[1] << ", " << devProp.maxGridSize[2] << ")" << std::endl;
        std::cout << "  Memory Clock Rate: " << devProp.memoryClockRate / 1000 << " MHz" << std::endl;
        std::cout << "  Memory Bus Width: " << devProp.memoryBusWidth << "-bit" << std::endl;
        std::cout << "  Peak Memory Bandwidth: " << 2.0 * devProp.memoryClockRate * (devProp.memoryBusWidth / 8) / 1.0e6 << " GB/s" << std::endl;
        std::cout << "  Total constant memory: " << devProp.totalConstMem / 1024 << " KB" << std::endl;
        std::cout << "  L2 Cache Size: " << devProp.l2CacheSize / 1024 << " KB" << std::endl;
        std::cout << "  Multiprocessor count: " << devProp.multiProcessorCount << std::endl;
        std::cout << "  Max threads per multiprocessor: " << devProp.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "  GPU clock rate: " << devProp.clockRate / 1000 << " MHz" << std::endl;
        std::cout << "  Memory pitch: " << devProp.memPitch / 1024 << " KB" << std::endl;
        std::cout << "  Texture alignment: " << devProp.textureAlignment << " bytes" << std::endl;
        std::cout << "  Concurrent kernels: " << (devProp.concurrentKernels ? "Yes" : "No") << std::endl;
        std::cout << "  ECC enabled: " << (devProp.ECCEnabled ? "Yes" : "No") << std::endl;
        std::cout << "  PCI bus ID: " << devProp.pciBusID << std::endl;
        std::cout << "  PCI device ID: " << devProp.pciDeviceID << std::endl;
        std::cout << "  PCI domain ID: " << devProp.pciDomainID << std::endl;
        std::cout << "  TCC driver: " << (devProp.tccDriver ? "Yes" : "No") << std::endl;
        std::cout << "  Async engine count: " << devProp.asyncEngineCount << std::endl;
        std::cout << "  Unified addressing: " << (devProp.unifiedAddressing ? "Yes" : "No") << std::endl;
        std::cout << "  Can map host memory: " << (devProp.canMapHostMemory ? "Yes" : "No") << std::endl;
        std::cout << "  Kernel exec timeout enabled: " << (devProp.kernelExecTimeoutEnabled ? "Yes" : "No") << std::endl;
        std::cout << "  Integrated GPU: " << (devProp.integrated ? "Yes" : "No") << std::endl;
        std::cout << "  Compute mode: " << devProp.computeMode << std::endl;
        std::cout << "--------------------------------" << std::endl;
    }
}
