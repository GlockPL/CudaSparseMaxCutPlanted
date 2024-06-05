#include "SparseMatrixSumKernel.cuh"


/// <summary>
/// This function sums either rows for matrix in csc format or columns for matrix in csr format
/// </summary>
/// <param name="nnz">Number on non zero elements</param>
/// <param name="d_non_offset_axis_ind">Array with nnz elements that is NOT in compressed format</param>
/// <param name="d_vals">Array with values</param>
/// <param name="d_axis_sum">Output array that will contain output of summed rows/cols</param>
/// <returns></returns>
__global__ void sum_axis(int nnz, const int* d_non_offset_axis_ind, const float* d_vals, float* d_axis_sum) {
    int element_ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (element_ind < nnz) {
        int idx = d_non_offset_axis_ind[element_ind];        
        atomicAdd(&d_axis_sum[idx], d_vals[idx]);
    }
}