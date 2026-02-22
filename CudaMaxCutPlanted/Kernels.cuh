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
// Initializes the random states for the kernels
__global__ void setup_kernel(unsigned long long seed, curandState* states);
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
// Converts on device char vector to float vector
__global__ void char_to_float(const char* input, float* output, int n);
// Encodes (I, J) pairs into a single int64 key (I*n + J) for single-pass sort
__global__ void make_sort_keys(const int* I, const int* J, int n, long long* keys, int total_n);
// Computes new_offsets[i] = old_offsets[i] + i (for inserting one entry per row)
__global__ void shift_offsets(const int* old_offsets, int* new_offsets, int n);
// Inserts one diagonal entry per row into a sorted CSR matrix (no existing diagonal assumed)
__global__ void insert_diagonal_csr(const int* old_offsets, const int* old_cols, const float* old_vals,
    const int* new_offsets, int* new_cols, float* new_vals, const float* diagonal, int n);
#endif // KERNELS_CUH