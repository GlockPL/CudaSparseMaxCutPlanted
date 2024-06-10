#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#ifdef __INTELLISENSE__
#include "intellisense_cuda_intrinsics.h"
#endif
#define BLOCK_SIZE 512


// Kernel to sum all elements along axis
__global__ void sum_axis(int nnz, const int* d_non_offset_axis_ind, const float* d_vals, float* d_axis_sum);
// Fills sparse graph with random values
__global__ void create_random_matrix(int n, int nnz, int split, const int* p, int* d_rows, int* d_cols, float* d_vals, curandState* states);
//Sets true/1 to the zereos vector x in positions provided by p
__global__ void set_true_elements(int split, const int* p, char* x);
//Count all non zero elements on the diagonal and returns bool vector where true is set on a position where diagonal is non zero
__global__ void non_zero_elements(const int* I, const int* J, bool* non_zero_elements, int* nnz_sum, int n);
// Sets values to the diagonal taking into account that diagonal can have non zero values
__global__ void set_diagonal(int* I, int* J, float* V, bool* non_zero_elements, const float* diagonal, int initial_n, int resize_n);
// Count and sets bool vector for zero elements inside float device vector
__global__ void zero_elements(const float* input_vect, bool* zero_elements_vect, int* zero_sum, int n);
#endif // KERNELS_CUH