#include "Kernels.cuh"

#define NUM_BANKS 16
#define LOG_NUM_BANKS 4
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))

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

__global__ void create_random_matrix(int n, int nnz, int split, const int* p, int* d_rows, int* d_cols, float* d_vals, curandState* states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nnz) {
        curand_init(1234, idx, 0, &states[idx]);
        int i = static_cast<int>(floorf(idx / (n - split)));
        int j = static_cast<int>((idx % (n - split)) + split);
        d_rows[idx] = p[i];
        d_cols[idx] = p[j];
        d_vals[idx] = curand_uniform(&states[idx]) * 0.99f + 0.01f;
    }
}

__global__ void set_true_elements(int split, const int* p, char* x) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < split) {
        x[p[idx]] = 1;
    }
}

__global__ void non_zero_elements(const int* I, const int* J, bool* non_zero_elements, int *nnz_sum, int nnz) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nnz) {
        if (I[idx] == J[idx]) {
            atomicAdd(nnz_sum, 1);
            non_zero_elements[idx] == true;
        }
        
    }
}

__global__ void zero_elements(const float *input_vect, bool* zero_elements_vect, int* zero_sum, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (fabsf(input_vect[idx]) < 1e-6) {
            atomicAdd(zero_sum, 1);
            zero_elements_vect[idx] = true;
        }

    }
}


__global__ void set_diagonal(int* I, int* J, float* V, bool* non_zero_elements, const float* diagonal, int initial_n, int resize_n) {
    int offset = initial_n;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < initial_n) {
        if (I[idx] == J[idx] && non_zero_elements[idx]) {
            
            V[idx] = diagonal[idx];
        }

    }
    else if (idx >= initial_n && idx < (initial_n + resize_n)) {
        int index = idx - offset;
        V[idx] = diagonal[index];
        I[idx] = index;
        J[idx] = index;
    }
}

__global__ void prescan(float* g_odata, float* g_idata, int n) {
    extern __shared__ float temp[];

    int thid = threadIdx.x;
    int offset = 1;

    // Load input into shared memory with padding to avoid bank conflicts
    int ai = thid;
    int bi = thid + (n / 2);
    int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

    temp[ai + bankOffsetA] = g_idata[ai];
    temp[bi + bankOffsetB] = g_idata[bi];

    // Build sum in place up the tree
    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (thid < d) {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    // Clear the last element
    if (thid == 0) {
        temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;
    }

    // Traverse down tree & build scan
    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (thid < d) {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    // Write results to device memory with proper offsets
    g_odata[ai] = temp[ai + bankOffsetA];
    g_odata[bi] = temp[bi + bankOffsetB];
}


