#ifndef SPARSEMATRIXSUMKERNEL_CUH
#define SPARSEMATRIXSUMKERNEL_CUH

#include <cuda_runtime.h>
#ifdef __INTELLISENSE__
#include "intellisense_cuda_intrinsics.h"
#endif


// Kernel to mark zero elements
__global__ void sum_axis(int nnz, const int* d_non_offset_axis_ind, const float* d_vals, float* d_axis_sum);

#endif // SPARSEMATRIXSUMKERNEL_CUH