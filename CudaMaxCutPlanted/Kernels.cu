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
// Requires d_non_offset_axis_ind to be sorted (ascending).
// Each warp performs a segmented inclusive scan over its 32 lanes: lanes with the
// same key accumulate into the rightmost lane of that segment, which then issues
// a single atomicAdd — reducing global atomic traffic by up to 32x.
__global__ void sum_axis(int nnz, const int* d_non_offset_axis_ind, const float* d_vals, float* d_axis_sum) {
    int idx  = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;

    float val = (idx < nnz) ? d_vals[idx]                : 0.0f;
    int   key = (idx < nnz) ? d_non_offset_axis_ind[idx] : -1;

    // Warp-level segmented inclusive scan (prefix sum within each key run).
    // Each step reads the value/key from `offset` lanes below; if the keys match,
    // the current lane folds that partial sum into its own accumulator.
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        float n_val = __shfl_up_sync(0xFFFFFFFF, val, offset);
        int   n_key = __shfl_up_sync(0xFFFFFFFF, key, offset);
        if (lane >= offset && n_key == key) val += n_val;
    }

    // A lane is a segment tail if the next lane carries a different key.
    // Inactive lanes (idx >= nnz) have key == -1, so they naturally terminate
    // any segment that ends at the last active lane.
    int  next_key = __shfl_down_sync(0xFFFFFFFF, key, 1);
    bool is_tail  = (idx < nnz) && (lane == 31 || next_key != key);

    if (is_tail) {
        atomicAdd(&d_axis_sum[key], val);
    }
}

__global__ void setup_kernel(unsigned long long seed, curandState* states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &states[idx]);
}

__global__ void create_random_matrix(int n, int nnz, int split, const int* p, int* d_rows, int* d_cols, float* d_vals, curandState* states) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = gridDim.x * blockDim.x;
    int half_nnz = nnz / 2;

    curandState localState = states[thread_id];

    for (int idx = thread_id; idx < half_nnz; idx += num_threads) {
        // Calculate i and j more safely
        int edges_per_first_node = n - split;
        int i = idx / edges_per_first_node;  // First partition node index
        int j = (idx % edges_per_first_node) + split;  // Second partition node index
        
        // Additional bounds checking
        if (i >= split || j >= n || j < split) continue;
        
        float weight = curand_uniform(&localState) * 0.99f + 0.01f;
        
        // Create edge (p[i], p[j])
        d_rows[idx] = p[i];
        d_cols[idx] = p[j];
        d_vals[idx] = weight;
        
        // Create symmetric edge (p[j], p[i]) with same weight
        d_rows[idx + half_nnz] = p[j];
        d_cols[idx + half_nnz] = p[i];
        d_vals[idx + half_nnz] = weight;
    }

    states[thread_id] = localState;
}

__global__ void set_true_elements(int split, const int* p, char* x) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < split) {
        x[p[idx]] = 1;
    }
}

__global__ void non_zero_elements(const int* I, const int* J, bool* non_zero_elements, int *nnz_sum, int nnz) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    bool is_diag = (idx < nnz) && (I[idx] == J[idx]);

    if (is_diag) {
        non_zero_elements[idx] = true;
    }

    // Warp-level reduction: count diagonal hits across 32 lanes with a single ballot
    unsigned int ballot = __ballot_sync(0xFFFFFFFF, is_diag);
    // Only lane 0 of each warp issues one atomicAdd with the warp's total count
    if ((threadIdx.x & 31) == 0) {
        atomicAdd(nnz_sum, __popc(ballot));
    }
}

__global__ void zero_elements(const float *input_vect, bool* zero_elements_vect, int* zero_sum, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    bool is_zero = (idx < n) && (fabsf(input_vect[idx]) < 1e-6f);

    if (is_zero) {
        zero_elements_vect[idx] = true;
    }

    unsigned int ballot = __ballot_sync(0xFFFFFFFF, is_zero);
    if ((threadIdx.x & 31) == 0) {
        atomicAdd(zero_sum, __popc(ballot));
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

__global__ void char_to_float(const char* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = static_cast<float>(input[idx]);
    }
}

// Encodes (I[idx], J[idx]) into a single int64 sort key = I*n + J.
// Allows a single sort pass instead of two-pass stable sort, using
// radix sort internally which needs far less temporary memory.
__global__ void make_sort_keys(const int* I, const int* J, int n, long long* keys, int total_n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_n) {
        keys[idx] = (long long)I[idx] * n + J[idx];
    }
}

// Computes new_offsets[i] = old_offsets[i] + i for i in [0, n].
// Used when inserting exactly one new entry per row (e.g. a diagonal entry)
// into a CSR matrix: each row i has i prior insertions before it.
__global__ void shift_offsets(const int* old_offsets, int* new_offsets, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i <= n) {
        new_offsets[i] = old_offsets[i] + i;
    }
}

// Inserts one diagonal entry (row, row, diagonal[row]) per row into a CSR matrix.
// Assumes no diagonal entry already exists in any row (columns are sorted ascending).
// One thread per row; new_offsets must be pre-computed via shift_offsets.
__global__ void insert_diagonal_csr(
    const int* old_offsets, const int* old_cols, const float* old_vals,
    const int* new_offsets, int* new_cols, float* new_vals,
    const float* diagonal, int n)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;

    int old_start = old_offsets[row];
    int old_end   = old_offsets[row + 1];
    int new_start = new_offsets[row];

    // Copy columns that come before the diagonal (col < row)
    int left = 0;
    while (old_start + left < old_end && old_cols[old_start + left] < row) {
        new_cols[new_start + left] = old_cols[old_start + left];
        new_vals[new_start + left] = old_vals[old_start + left];
        left++;
    }

    // Insert diagonal entry
    new_cols[new_start + left] = row;
    new_vals[new_start + left] = diagonal[row];

    // Copy columns that come after the diagonal (col > row)
    int old_right = old_start + left;
    int new_right = new_start + left + 1;
    for (int k = old_right; k < old_end; k++) {
        new_cols[new_right + (k - old_right)] = old_cols[k];
        new_vals[new_right + (k - old_right)] = old_vals[k];
    }
}
