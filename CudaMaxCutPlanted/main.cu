#include "CudaSparseMatrix.hpp"
#include <cublas_v2.h>
#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
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
    int half_nnz = nnz / 2;  // Each thread creates 2 symmetric edges
    CHECK_CUDA(cudaMalloc((void**)&states, half_nnz * sizeof(states)));
    int gridSize = (half_nnz + BLOCK_SIZE - 1) / BLOCK_SIZE;

    create_random_matrix << <gridSize, BLOCK_SIZE >> > (n, nnz, split, p, I, J, V, states);

    CHECK_CUDA(cudaDeviceSynchronize());

    // Wrap raw device pointers in thrust device pointers for sorting
    thrust::device_ptr<int> dev_I(I);
    thrust::device_ptr<int> dev_J(J);
    thrust::device_ptr<float> dev_V(V);

    // First, sort by the secondary key (J) using stable sort
    thrust::stable_sort_by_key(dev_J, dev_J + nnz, thrust::make_zip_iterator(thrust::make_tuple(dev_I, dev_V)));

    // Then, sort by the primary key (I) using stable sort to maintain the order of the secondary key
    thrust::stable_sort_by_key(dev_I, dev_I + nnz, thrust::make_zip_iterator(thrust::make_tuple(dev_J, dev_V)));


    int* h_rows = new int[nnz];
    int* h_cols = new int[nnz];
    float* h_vals = new float[nnz];
    

    CHECK_CUDA(cudaMemcpy(h_rows, I, nnz * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_cols, J, nnz * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_vals, V, nnz * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < nnz; ++i) {
        std::cout << "I: " << h_rows[i] << " J: " << h_cols[i] << " V: " << h_vals[i] << std::endl;
    }

    CHECK_CUDA(cudaFree(states));
    delete[] h_rows;
    delete[] h_cols;
    delete[] h_vals;   
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
    float* row_sum = Q.sum(0);
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


int main() {
    int n = 20;
    int seed = 14;
    float density = 0.5;
    int* I, * J;
    float* V;
    std::mt19937 rng(seed);

    int* p = generate_initial_permutation(rng, n);

    int split = estimate_split(density, n);  // Example split, can be computed as needed
    int nnz = 2 * split * (n - split);  // Double for symmetric graph
    std::cout << "Split: " << split << " nnz (symmetric): " << nnz << std::endl;

    CHECK_CUDA(cudaMalloc((void**)&I, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&J, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&V, nnz * sizeof(float)));

    create_graph_sparse(n, nnz, split, p, I, J, V);

    CudaSparseMatrix graph = CudaSparseMatrix(I, J, V, n, nnz, SparseType::COO, MemoryType::Device);
    graph.display();

    char* x = generate_solution(p, split, n);

    char* h_x = new char[n];

    CHECK_CUDA(cudaMemcpy(h_x, x, n * sizeof(char), cudaMemcpyDeviceToHost));

    for (int i = 0; i < n; i++)
    {
        std::cout << "X_" << i << " " << static_cast<int>(h_x[i]) << std::endl;
    }

    CudaSparseMatrix Q = CudaSparseMatrix(graph);
    graph.clear();

    graph_to_qubo(Q);

    Q.display();

    // Calculate QUBO energy of the planted solution
    float energy = calculate_qubo_energy(Q, x);
    std::cout << "QUBO Energy of planted solution: " << energy << std::endl;

    // Run brute force solution finder to verify optimality
    std::cout << "\n=== Starting Brute Force Verification ===" << std::endl;
    brute_force_solution_finder(Q, h_x);

    cudaFree(I);
    cudaFree(J);
    cudaFree(V);
    cudaFree(p);
    cudaFree(x);

    delete[] h_x;

    return 0;
};
