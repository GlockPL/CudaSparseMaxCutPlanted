#include "CudaSparseMatrix.hpp"
#include "device_properties.cuh"
#include <cublas_v2.h>
#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/gather.h>
#include <thrust/sequence.h>
#include <random>
#include <curand_kernel.h>
#include <limits>
#include <cstring>
#include <iomanip>
#include "Kernels.cuh"
#include "indicators.hpp"
#ifdef __INTELLISENSE__
#include "intellisense_cuda_intrinsics.h"
#endif

#define CHECK_CUBLAS(call)                                        \
{                                                                 \
    cublasStatus_t err = call;                                    \
    if (err != CUBLAS_STATUS_SUCCESS) {                           \
        std::cerr << "CUBLAS error in file " << __FILE__          \
                  << " at line " << __LINE__ << ": "            \
                  << "Error code " << err << std::endl;      \
        exit(EXIT_FAILURE);                                       \
    }                                                             \
}

#define CUDA_PRINT_MEM(label) {                               \
      size_t _free, _total;                                     \
      cudaMemGetInfo(&_free, &_total);                          \
      printf("[MEM] %-40s  used=%6zu MB  free=%6zu MB\n",      \
          label,                                                \
          (_total-_free)/(1024*1024), _free/(1024*1024));       \
  }



void scale_csr_matrix(cusparseHandle_t handle,
    float alpha,
    cusparseSpMatDescr_t& input,
    cusparseSpMatDescr_t& result) {
    // Extract the dimensions and the number of non-zero elements
    int64_t n, nnz;
    int* d_csrOffsets;
    int* d_cols;
    float* d_vals;

    int64_t n_r, nnz_r;
    int* d_csrOffsets_r;
    int* d_cols_r;
    float* d_vals_r;

    cusparseIndexType_t csrRowOffsetsType;
    cusparseIndexType_t csrColIndType;
    cusparseIndexBase_t idxBase;
    cudaDataType valueTyp;

    cusparseCsrGet(input, &n, &n, &nnz, (void**)&d_csrOffsets, (void**)&d_cols, (void**)&d_vals, &csrRowOffsetsType, &csrColIndType, &idxBase, &valueTyp);
    cusparseCsrGet(result, &n_r, &n_r, &nnz_r, (void**)&d_csrOffsets_r, (void**)&d_cols_r, (void**)&d_vals_r, &csrRowOffsetsType, &csrColIndType, &idxBase, &valueTyp);

    // Set scaling factors
    const float beta = 0.0f;
    size_t bufferSize = 0;

    cusparseMatDescr_t input_desc;
    cusparseCreateMatDescr(&input_desc);

    // Create matrix descriptor for the result matrix C
    cusparseMatDescr_t result_desc;
    cusparseCreateMatDescr(&result_desc);

    // Get buffer size for the operation
    cusparseScsrgeam2_bufferSizeExt(handle,
        n, n,
        &alpha, input_desc, nnz,
        d_vals,
        d_csrOffsets,
        d_cols,
        &beta, input_desc, nnz,
        d_vals,
        d_csrOffsets,
        d_cols,
        result_desc,
        d_vals_r,
        d_csrOffsets_r,
        d_cols_r,
        &bufferSize);

    void* dBuffer;
    cudaMalloc(&dBuffer, bufferSize);

    // Perform the scaling operation
    cusparseScsrgeam2(handle,
        n, n,
        &alpha, input_desc, nnz,
        d_vals,
        d_csrOffsets,
        d_cols,
        &beta, input_desc, nnz,
        d_vals,
        d_csrOffsets,
        d_cols,
        result_desc,
        d_vals_r,
        d_csrOffsets_r,
        d_cols_r,
        dBuffer);
    // Clean up
    cudaFree(dBuffer);
    cusparseDestroyMatDescr(input_desc);
    cusparseDestroyMatDescr(result_desc);
}

// Function to generate initial permutation
int* generate_initial_permutation(std::mt19937& rng, int n) {
    std::vector<int> permutation(n);
    for (int i = 0; i < n; ++i) {
        permutation[i] = i;
    }
    std::shuffle(permutation.begin(), permutation.end(), rng);

    int* p;
    CHECK_CUDA(cudaMalloc((void**)&p, n * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(p, permutation.data(), n * sizeof(int), cudaMemcpyHostToDevice));

    return p;
}


void create_graph_sparse(int n, int nnz, int split, const int* p, int *I, int* J, float* V) {

    curandState* states;
    int half_nnz = nnz / 2;
    
    // We want to launch enough threads to keep the GPU busy, but not an excessive amount.
    // Let's aim for a high number of threads, e.g., 256*1024, but not more than half_nnz.
    int num_threads_to_launch = std::min(half_nnz, 256 * 1024);
    int gridSize = (num_threads_to_launch + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int num_threads = gridSize * BLOCK_SIZE;

    std::cout << "Kernel launch parameters: half_nnz=" << half_nnz << ", split=" << split << ", n-split=" << (n-split) << std::endl;
    std::cout << "Launching " << num_threads << " threads in " << gridSize << " blocks." << std::endl;

    // Validate expected number of edges
    long long expected_edges = (long long)split * (n - split);
    if (half_nnz != expected_edges) {
        std::cerr << "WARNING: half_nnz (" << half_nnz << ") != expected_edges (" << expected_edges << ")" << std::endl;
    }
    
    CHECK_CUDA(cudaMalloc((void**)&states, num_threads * sizeof(curandState)));
    
    // Initialize random states
    setup_kernel<<<gridSize, BLOCK_SIZE>>>(1234, states);
    CHECK_CUDA(cudaGetLastError());
    CUDA_PRINT_MEM("After setup kernel");
    // Create the random matrix
    create_random_matrix<<<gridSize, BLOCK_SIZE>>>(n, nnz, split, p, I, J, V, states);
    CHECK_CUDA(cudaGetLastError());
    CUDA_PRINT_MEM("After create random matrix");
    CHECK_CUDA(cudaDeviceSynchronize());

    // Single-pass sort using a combined int64 key (I*n + J).
    // Avoids two passes of stable_sort_by_key (merge sort, ~2x temp per pass).
    long long* d_sort_keys;
    CHECK_CUDA(cudaMalloc((void**)&d_sort_keys, nnz * sizeof(long long)));

    int gridSizeKeys = (nnz + BLOCK_SIZE - 1) / BLOCK_SIZE;
    make_sort_keys<<<gridSizeKeys, BLOCK_SIZE>>>(I, J, n, d_sort_keys, nnz);
    CHECK_CUDA(cudaGetLastError());
    CUDA_PRINT_MEM("After make sort keys");

    // Sort a permutation array by the combined key.
    // sort_by_key(int64, int) uses CUB radix sort — temp = nnz*12 bytes (one output copy
    // of keys + values). The zip approach used merge sort needing ~2x full data = OOM.
    int* d_perm;
    CHECK_CUDA(cudaMalloc((void**)&d_perm, nnz * sizeof(int)));
    thrust::sequence(thrust::device_ptr<int>(d_perm), thrust::device_ptr<int>(d_perm + nnz));

    thrust::device_ptr<long long> dev_keys(d_sort_keys);
    thrust::sort_by_key(dev_keys, dev_keys + nnz, thrust::device_ptr<int>(d_perm));
    CHECK_CUDA(cudaFree(d_sort_keys));
    CUDA_PRINT_MEM("After sorting keys");

    // Gather I, J, V in sorted order one array at a time to keep peak memory low.
    // Peak per gather: I+J+V (live) + perm + one temp = 5x nnz*4 bytes.
    int* d_temp_int;
    CHECK_CUDA(cudaMalloc((void**)&d_temp_int, nnz * sizeof(int)));

    thrust::gather(thrust::device_ptr<int>(d_perm), thrust::device_ptr<int>(d_perm + nnz),
        thrust::device_ptr<int>(I), thrust::device_ptr<int>(d_temp_int));
    CHECK_CUDA(cudaMemcpy(I, d_temp_int, nnz * sizeof(int), cudaMemcpyDeviceToDevice));

    thrust::gather(thrust::device_ptr<int>(d_perm), thrust::device_ptr<int>(d_perm + nnz),
        thrust::device_ptr<int>(J), thrust::device_ptr<int>(d_temp_int));
    CHECK_CUDA(cudaMemcpy(J, d_temp_int, nnz * sizeof(int), cudaMemcpyDeviceToDevice));

    CHECK_CUDA(cudaFree(d_temp_int));

    float* d_temp_float;
    CHECK_CUDA(cudaMalloc((void**)&d_temp_float, nnz * sizeof(float)));
    thrust::gather(thrust::device_ptr<int>(d_perm), thrust::device_ptr<int>(d_perm + nnz),
        thrust::device_ptr<float>(V), thrust::device_ptr<float>(d_temp_float));
    CHECK_CUDA(cudaMemcpy(V, d_temp_float, nnz * sizeof(float), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaFree(d_temp_float));

    CHECK_CUDA(cudaFree(d_perm));
    CUDA_PRINT_MEM("After gather");

    CHECK_CUDA(cudaFree(states));
}

char* generate_solution(const int* p, int split, int n) {
    char* x;
    CHECK_CUDA(cudaMalloc((void**)&x, n * sizeof(char)));
    // Initialize a device vector of bool type with false 
    CHECK_CUDA(cudaMemset(x, 0, n));
    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    set_true_elements << <gridSize, BLOCK_SIZE >> > (split, p, x);
    CHECK_CUDA(cudaDeviceSynchronize());

    return x;
}

void graph_to_qubo(CudaSparseMatrix& Q) {
    float* row_sum = Q.sum(1);
    Q.multiply(-1.0f);
    Q.fill_diagonal(row_sum);
    Q.multiply(-0.25f);
}

float calculate_qubo_energy(const CudaSparseMatrix& Q, const char* x) {
    int n = Q.size();
    
    // Convert char* x to float* for matrix operations
    float* x_float;
    CHECK_CUDA(cudaMalloc((void**)&x_float, n * sizeof(float)));
    
    // Launch kernel to convert char to float
    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    char_to_float<<<gridSize, BLOCK_SIZE>>>(x, x_float, n);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Calculate Q * x using the CudaSparseMatrix dot method
    CudaDenseVector Qx = Q.dot(x_float);
    
    // Create CudaDenseVector from x_float for dot product
    CudaDenseVector x_vector(n, x_float, MemoryType::Device);
    
    // Calculate x^T * (Q * x) using CudaDenseVector dot method
    float result = x_vector.dot(Qx);
    
    // Clean up
    CHECK_CUDA(cudaFree(x_float));
    
    return result;
}

void brute_force_solution_finder(const CudaSparseMatrix& Q, const char* planted_solution) {
    int n = Q.size();
    int num_solutions = 1 << n;  // 2^n possible binary vectors
    
    std::cout << "Starting brute force search over " << num_solutions << " possible solutions..." << std::endl;
    
    float min_energy = std::numeric_limits<float>::max();
    float planted_energy = 0.0f;
    int best_solution_index = -1;
    char* best_solution = new char[n];
    char* current_solution = new char[n];
    
    // Allocate device memory for current solution
    char* d_current_solution;
    CHECK_CUDA(cudaMalloc((void**)&d_current_solution, n * sizeof(char)));
    
    // Progress tracking
    int progress_step = std::max(1, num_solutions / 100);  // Update every 1%
    
    // Test all possible binary vectors
    for (int i = 0; i < num_solutions; ++i) {
        // Generate binary vector from integer i
        for (int j = 0; j < n; ++j) {
            current_solution[j] = (i & (1 << j)) ? 1 : 0;
        }
        
        // Copy to device
        CHECK_CUDA(cudaMemcpy(d_current_solution, current_solution, n * sizeof(char), cudaMemcpyHostToDevice));
        
        // Calculate energy
        float energy = calculate_qubo_energy(Q, d_current_solution);
        
        // Check if this is the best solution so far
        if (energy < min_energy) {
            min_energy = energy;
            best_solution_index = i;
            std::memcpy(best_solution, current_solution, n * sizeof(char));
        }
        
        // Check if this matches the planted solution
        bool is_planted = true;
        for (int j = 0; j < n; ++j) {
            if (current_solution[j] != planted_solution[j]) {
                is_planted = false;
                break;
            }
        }
        
        if (is_planted) {
            planted_energy = energy;
            std::cout << "Found planted solution at index " << i << " with energy: " << planted_energy << std::endl;
        }
        
        // Progress update
        if (i % progress_step == 0) {
            float progress = (float)i / num_solutions * 100.0f;
            std::cout << "Progress: " << std::fixed << std::setprecision(1) << progress 
                     << "% (tested " << i << "/" << num_solutions << " solutions)" << std::endl;
        }
    }
    
    // Results
    std::cout << "\n=== BRUTE FORCE RESULTS ===" << std::endl;
    std::cout << "Total solutions tested: " << num_solutions << std::endl;
    std::cout << "Best solution found at index: " << best_solution_index << std::endl;
    std::cout << "Best energy: " << min_energy << std::endl;
    std::cout << "Planted solution energy: " << planted_energy << std::endl;
    
    std::cout << "\nBest solution vector: ";
    for (int i = 0; i < n; ++i) {
        std::cout << static_cast<int>(best_solution[i]) << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Planted solution vector: ";
    for (int i = 0; i < n; ++i) {
        std::cout << static_cast<int>(planted_solution[i]) << " ";
    }
    std::cout << std::endl;
    
    // Check if planted solution is optimal
    if (std::abs(planted_energy - min_energy) < 1e-6) {
        std::cout << "\n✓ SUCCESS: Planted solution is OPTIMAL!" << std::endl;
    } else {
        std::cout << "\n✗ FAILURE: Planted solution is NOT optimal." << std::endl;
        std::cout << "Energy gap: " << (planted_energy - min_energy) << std::endl;
    }
    
    // Cleanup
    CHECK_CUDA(cudaFree(d_current_solution));
    delete[] best_solution;
    delete[] current_solution;
}

int estimate_split(float density, int n) {
    if (density > 0.5f) {
        std::cout << "Error, density can not be bigger than 0.5!" << std::endl;
    }
    
    float sparsity = 1.0f - density;
    float inside = 2.0f * sparsity - 1.0f;
    float smaller_set_size = 0.5f * n * (1 + sqrt(inside));
    return static_cast<int>(smaller_set_size) - 1;
}


int main(int argc, char* argv[]) {
    printGpuProperties();
    int n = 60000; // Default value

    // --- Argument Parsing ---
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-n") {
            if (i + 1 < argc) { // Make sure we aren't at the end of argv
                try {
                    n = std::stoi(argv[++i]);
                }
                catch (const std::invalid_argument& ia) {
                    std::cerr << "Invalid argument: -n must be followed by an integer." << std::endl;
                    return 1;
                }
                catch (const std::out_of_range& oor) {
                    std::cerr << "Argument out of range for -n." << std::endl;
                    return 1;
                }
            } else {
                std::cerr << "-n option requires one argument." << std::endl;
                return 1;
            }
        }
    }
    // --- End Argument Parsing ---

    int seed = 8848;
    float density = 0.1;
    int* I, * J;
    float* V;
    std::mt19937 rng(seed);

    int* p = generate_initial_permutation(rng, n);
    CUDA_PRINT_MEM("After intial permutation");
    int split = estimate_split(density, n);  // Example split, can be computed as needed
    
    // Check for integer overflow in nnz calculation
    long long nnz_calc = 2LL * split * (n - split);
    if (nnz_calc > INT_MAX) {
        std::cerr << "ERROR: nnz calculation exceeds INT_MAX: " << nnz_calc << std::endl;
        return -1;
    }
    int nnz = static_cast<int>(nnz_calc);
    std::cout << "Matrix width: " << n << std::endl;
    std::cout << "Split: " << split << " nnz (symmetric): " << nnz << std::endl;
    
    // Check GPU memory availability
    size_t free_mem, total_mem;
    CHECK_CUDA(cudaMemGetInfo(&free_mem, &total_mem));
    size_t required_mem = nnz * (2 * sizeof(int) + sizeof(float)) + n * sizeof(int);
    std::cout << "GPU Memory - Total: " << total_mem / (1024*1024) << " MB, ";
    std::cout << "Free: " << free_mem / (1024*1024) << " MB, ";
    std::cout << "Required: " << required_mem / (1024*1024) << " MB" << std::endl;
    
    if (required_mem > free_mem) {
        std::cerr << "ERROR: Not enough GPU memory! Required: " << required_mem / (1024*1024) 
                  << " MB, Available: " << free_mem / (1024*1024) << " MB" << std::endl;
        return -1;
    }

    CHECK_CUDA(cudaMalloc((void**)&I, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&J, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&V, nnz * sizeof(float)));

    create_graph_sparse(n, nnz, split, p, I, J, V);
    CUDA_PRINT_MEM("after create_graph_sparse");
    CudaSparseMatrix graph = CudaSparseMatrix(I, J, V, n, nnz, SparseType::COO, MemoryType::Device);
    CUDA_PRINT_MEM("after graph construct");
    cudaFree(I);
    cudaFree(J);
    cudaFree(V);
    I = J = nullptr; V = nullptr;
    
    // graph.display();

    char* x = generate_solution(p, split, n);

    char* h_x = new char[n];

    CHECK_CUDA(cudaMemcpy(h_x, x, n * sizeof(char), cudaMemcpyDeviceToHost));

    // Only print first 10 elements for large matrices
    int print_limit = std::min(n, 10);
    for (int i = 0; i < print_limit; i++)
    {
        std::cout << "X_" << i << " " << static_cast<int>(h_x[i]) << std::endl;
    }
    if (n > 10) {
        std::cout << "... (" << (n - 10) << " more elements)" << std::endl;
    }

    CudaSparseMatrix Q = CudaSparseMatrix(graph);
    graph.clear();

    graph_to_qubo(Q);

    // Q.display();

    // Calculate QUBO energy of the planted solution
    float energy = calculate_qubo_energy(Q, x);
    std::cout << "QUBO Energy of planted solution: " << energy << std::endl;

    // Run brute force solution finder to verify optimality
    // std::cout << "\n=== Starting Brute Force Verification ===" << std::endl;
    // brute_force_solution_finder(Q, h_x);

    
    cudaFree(p);
    cudaFree(x);

    delete[] h_x;

    return 0;
};
