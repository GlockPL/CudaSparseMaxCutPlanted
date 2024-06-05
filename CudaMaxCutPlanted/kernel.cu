#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/random.h>
#include <thrust/reduce.h>
#include <thrust/extrema.h>
#include <thrust/sequence.h>
#include <thrust/inner_product.h>
#include <iostream>
#include <algorithm>
#include <random>
#include <vector>
#include <iostream>
#include <iomanip>
#include "indicators.hpp"
#ifdef __INTELLISENSE__
#include "intellisense_cuda_intrinsics.h"
#endif


#define CHECK_CUSPARSE(call)                                    \
{                                                               \
    cusparseStatus_t err = call;                                \
    if (err != CUSPARSE_STATUS_SUCCESS) {                       \
        std::cerr << "CUSPARSE error in file " << __FILE__      \
                  << " at line " << __LINE__ << ": "            \
                  << cusparseGetErrorString(err) << std::endl;  \
        exit(EXIT_FAILURE);                                     \
    }                                                           \
}

#define CHECK_CUDA(call)                                        \
{                                                               \
    cudaError_t err = call;                                     \
    if (err != cudaSuccess) {                                   \
        std::cerr << "CUDA error in file " << __FILE__          \
                  << " at line " << __LINE__ << ": "            \
                  << cudaGetErrorString(err) << std::endl;      \
        exit(EXIT_FAILURE);                                     \
    }                                                           \
}

#define CHECK_CUBLAS(call)                                        \
{                                                                 \
    cublasStatus_t err = call;                                    \
    if (err != CUBLAS_STATUS_SUCCESS) {                           \
        std::cerr << "CUBLAS error in file " << __FILE__          \
                  << " at line " << __LINE__ << ": "              \
                  << cublasGetErrorString(err) << std::endl;      \
        exit(EXIT_FAILURE);                                       \
    }                                                             \
}


struct csr_data {
    int* rowPointer;
    int* cols;
    float* vals;
};

const char* cusparseGetErrorString(cusparseStatus_t status) {
    switch (status) {
    case CUSPARSE_STATUS_SUCCESS: return "CUSPARSE_STATUS_SUCCESS";
    case CUSPARSE_STATUS_NOT_INITIALIZED: return "CUSPARSE_STATUS_NOT_INITIALIZED";
    case CUSPARSE_STATUS_ALLOC_FAILED: return "CUSPARSE_STATUS_ALLOC_FAILED";
    case CUSPARSE_STATUS_INVALID_VALUE: return "CUSPARSE_STATUS_INVALID_VALUE";
    case CUSPARSE_STATUS_ARCH_MISMATCH: return "CUSPARSE_STATUS_ARCH_MISMATCH";
    case CUSPARSE_STATUS_MAPPING_ERROR: return "CUSPARSE_STATUS_MAPPING_ERROR";
    case CUSPARSE_STATUS_EXECUTION_FAILED: return "CUSPARSE_STATUS_EXECUTION_FAILED";
    case CUSPARSE_STATUS_INTERNAL_ERROR: return "CUSPARSE_STATUS_INTERNAL_ERROR";
    case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED: return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
    case CUSPARSE_STATUS_ZERO_PIVOT: return "CUSPARSE_STATUS_ZERO_PIVOT";
    default: return "UNKNOWN CUSPARSE STATUS";
    }
}

const char* cublasGetErrorString(cublasStatus_t status) {
    switch (status) {
    case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
    default: return "UNKNOWN CUBLAS STATUS";
    }
}

__global__ void cols_sum(int n, const int* d_csrOffsets, const float* d_vals, float* d_rowSums) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n) {
        int row_start = d_csrOffsets[row];
        int row_end = d_csrOffsets[row + 1];
        for (int j = row_start; j < row_end; j++) {
            atomicAdd(&d_rowSums[row], d_vals[j]);
        }
    }
}

void convert_sparse_to_dense_and_display(cusparseHandle_t handle, const cusparseSpMatDescr_t& matDescr, int n) {
    // Allocate memory for the dense matrix on the device
    float* d_denseMat;
    cudaMalloc((void**)&d_denseMat, n * n * sizeof(float));

    // Create a dense matrix descriptor
    cusparseDnMatDescr_t denseDescr;
    CHECK_CUSPARSE(cusparseCreateDnMat(&denseDescr,
        n, // number of rows
        n, // number of columns
        n, // leading dimension
        d_denseMat, // pointer to dense matrix data
        CUDA_R_32F, // data type
        CUSPARSE_ORDER_ROW)); // row-major order

    // Convert sparse matrix to dense matrix
    void* dBuffer = NULL;
    size_t bufferSize = 0;
    CHECK_CUSPARSE(cusparseSparseToDense_bufferSize(handle,
        matDescr,
        denseDescr,
        CUSPARSE_SPARSETODENSE_ALG_DEFAULT,
        &bufferSize));

    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

    CHECK_CUSPARSE(cusparseSparseToDense(handle,
        matDescr,
        denseDescr,
        CUSPARSE_SPARSETODENSE_ALG_DEFAULT,
        dBuffer));

    // Copy the dense matrix from device to host
    std::vector<float> h_denseMat(n * n);
    CHECK_CUDA(cudaMemcpy(h_denseMat.data(), d_denseMat, n * n * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << std::fixed << std::setprecision(4); // Set precision to 2 decimal places
    std::cout << "Dense matrix:" << std::endl;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << h_denseMat[i * n + j] << " ";
        }
        std::cout << std::endl;
    }

    // Clean up
    CHECK_CUDA(cudaFree(d_denseMat));
    CHECK_CUDA(cudaFree(dBuffer));
    CHECK_CUSPARSE(cusparseDestroyDnMat(denseDescr));
}


void fill_diagonal(cusparseHandle_t handle, cusparseSpMatDescr_t& input, thrust::device_vector<float> diag, csr_data& extended_pointers) {
    int64_t n, nnz;
    int* d_csrOffsets;
    int* d_cols;
    float* d_vals;

    cusparseIndexType_t csrRowOffsetsType;
    cusparseIndexType_t csrColIndType;
    cusparseIndexBase_t idxBase;
    cudaDataType valueTyp;

    CHECK_CUSPARSE(cusparseCsrGet(input, &n, &n, &nnz, (void**)&d_csrOffsets, (void**)&d_cols, (void**)&d_vals, &csrRowOffsetsType, &csrColIndType, &idxBase, &valueTyp));

    thrust::device_vector<int> d_rows_tdv(nnz);
    thrust::device_vector<int> d_cols_tdv(d_cols, d_cols+nnz);
    thrust::device_vector<float> d_vals_tdv(d_vals, d_vals+nnz);

    CHECK_CUSPARSE(cusparseXcsr2coo(handle,
        d_csrOffsets,
        nnz,
        n,
        thrust::raw_pointer_cast(d_rows_tdv.data()),
        CUSPARSE_INDEX_BASE_ZERO));

    d_rows_tdv.resize(nnz + n);
    d_cols_tdv.resize(nnz + n);
    d_vals_tdv.resize(nnz + n);

    int nnz_n = nnz + n;

    thrust::device_vector<int> d_vec(n);

    // Fill the vector with values from 0 to n-1
    thrust::sequence(d_vec.begin(), d_vec.end());

    thrust::copy(d_vec.begin(), d_vec.end(), d_rows_tdv.begin() + nnz);
    thrust::copy(d_vec.begin(), d_vec.end(), d_cols_tdv.begin() + nnz);
    thrust::copy(diag.begin(), diag.end(), d_vals_tdv.begin() + nnz);

    /*thrust::copy(d_rows_tdv.begin(), d_rows_tdv.end(), );
    thrust::copy(d_cols_tdv.begin(), d_cols_tdv.end(), extended_pointers.cols);
    thrust::copy(d_vals_tdv.begin(), d_vals_tdv.end(), extended_pointers.vals);*/


    CHECK_CUSPARSE(cusparseXcsr2coo(handle,
        d_csrOffsets,
        nnz,
        n,
        thrust::raw_pointer_cast(d_rows_tdv.data()),
        CUSPARSE_INDEX_BASE_ZERO));
    thrust::device_vector<int> d_csrOffsets_o(n + 1);

    thrust::sort_by_key(thrust::make_zip_iterator(thrust::make_tuple(d_rows_tdv.begin(), d_cols_tdv.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(d_rows_tdv.end(), d_cols_tdv.end())),
        d_vals_tdv.begin());
    
    CHECK_CUSPARSE(cusparseXcoo2csr(handle,
        thrust::raw_pointer_cast(d_rows_tdv.data()),
        nnz + n,
        n,
        thrust::raw_pointer_cast(d_csrOffsets_o.data()),
        CUSPARSE_INDEX_BASE_ZERO));

    CHECK_CUDA(cudaMemcpy(extended_pointers.rowPointer, thrust::raw_pointer_cast(d_csrOffsets_o.data()), (n + 1) * sizeof(int), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(extended_pointers.cols, thrust::raw_pointer_cast(d_cols_tdv.data()), nnz_n * sizeof(int), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(extended_pointers.vals, thrust::raw_pointer_cast(d_vals_tdv.data()), nnz_n * sizeof(float), cudaMemcpyDeviceToDevice));

    CHECK_CUSPARSE(cusparseCsrSetPointers(input,
        extended_pointers.rowPointer,
        extended_pointers.cols,
        extended_pointers.vals));

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

thrust::device_vector<float> sum_rows_csr_matrix(cusparseHandle_t handle, cusparseSpMatDescr_t input) {
    // Extract CSR matrix information
    int64_t rows, cols, nnz;
    int* d_csrOffsets, * d_cols;
    float* d_vals;

    cusparseIndexType_t csrRowOffsetsType;
    cusparseIndexType_t csrColIndType;
    cusparseIndexBase_t idxBase;
    cudaDataType valueTyp;

    cusparseCsrGet(input, &rows, &cols, &nnz,
        (void**)&d_csrOffsets, (void**)&d_cols, (void**)&d_vals,
        &csrRowOffsetsType, &csrColIndType, &idxBase, &valueTyp);

    // Allocate memory for the row sums
    thrust::device_vector<float> d_rowSums(rows);

    // Launch kernel to sum elements of each row using atomicAdd
    int blockSize = 256;
    int gridSize = (rows + blockSize - 1) / blockSize;
    cols_sum << <gridSize, blockSize >> > (rows, d_csrOffsets, d_vals, thrust::raw_pointer_cast(d_rowSums.data()));
    cudaDeviceSynchronize();

    return d_rowSums;
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

// Function to generate initial permutation
std::vector<int> generate_initial_permutation(std::mt19937 & rng, int n) {
    std::vector<int> permutation(n);
    for (int i = 0; i < n; ++i) {
        permutation[i] = i;
    }
    std::shuffle(permutation.begin(), permutation.end(), rng);
    return permutation;
}


void create_graph_sparse(int n, int split, std::vector<int>& p, thrust::device_vector<int>& d_rows, thrust::device_vector<int>& d_cols, thrust::device_vector<float>& d_vals) {
    thrust::host_vector<int> h_rows, h_cols;
    thrust::host_vector<float> h_vals;

    thrust::default_random_engine rng;
    thrust::random::uniform_real_distribution<float> dist(0.01f, 1.0f);

    std::vector<std::tuple<int, int, float>> combined;

    for (int i = 0; i <= split; ++i) {
        for (int j = split + 1; j < n; ++j) {
            float rnd_val = dist(rng);
            combined.push_back(std::make_tuple(p[i], p[j], rnd_val));
            combined.push_back(std::make_tuple(p[j], p[i], rnd_val));
        }
    }

    std::sort(combined.begin(), combined.end(), [](const auto& a, const auto& b) {
        if (std::get<0>(a) == std::get<0>(b)) {
            return std::get<1>(a) < std::get<1>(b);
        }
        return std::get<0>(a) < std::get<0>(b);
        });

    for (size_t i = 0; i < combined.size(); ++i) {
        h_rows.push_back(std::get<0>(combined[i]));
        h_cols.push_back(std::get<1>(combined[i]));
        h_vals.push_back(std::get<2>(combined[i]));
    }

    d_rows = h_rows;
    d_cols = h_cols;
    d_vals = h_vals;
}

thrust::device_vector<float> generate_solution(const std::vector<int>& p, int split) {
    int size = p.size();

    // Initialize a device vector of bool type with false (equivalent to CUDA.zeros(Bool, size(p)))
    thrust::device_vector<float> x(size, 0);

    
    // Set the first split elements to true (equivalent to x[p[1:split]] .= 1)
    for (int i = 0; i <= split ; ++i) {
        x[p[i]] = 1;
    }

    return x;
}


void graph_to_qubo(cusparseHandle_t handle, cusparseSpMatDescr_t& graph, cusparseSpMatDescr_t& Q, csr_data& extended_pointers) {
    scale_csr_matrix(handle, -1.0f, graph, Q);
    thrust::device_vector<float> sum = sum_rows_csr_matrix(handle, graph);
    fill_diagonal(handle, Q, sum, extended_pointers);
    scale_csr_matrix(handle, -0.25f, Q, Q);

    for (int i = 0; i < sum.size(); i++) {
        std::cout << sum[i] << std::endl;
    }
}

//float calculate_qubo_energy(cublasHandle_t cublasHandle,
//    cusparseHandle_t handle,
//    int n,
//    const cusparseSpMatDescr_t& Q,
//    const thrust::device_vector<float>& x) {
//    float alpha = 1.0f;
//    float beta = 0.0f;
//    size_t bufferSize = 0;
//    void* dBuffer = nullptr;
//
//    // Create dense vector descriptors
//    cusparseDnVecDescr_t vecX, vecY;
//    float* Qx;
//    float* in_x;
//
//    cudaMalloc((void**)&Qx, n * sizeof(float));
//    cudaMalloc((void**)&in_x, n * sizeof(float));
//
//    cudaMemcpy(in_x, thrust::raw_pointer_cast(x.data()), n * sizeof(float), cudaMemcpyDeviceToDevice);
//
//    cusparseCreateDnVec(&vecX, n, in_x, CUDA_R_32F);
//    cusparseCreateDnVec(&vecY, n, Qx, CUDA_R_32F);
//    // Allocate buffer
//    cusparseSpMV_bufferSize(handle,
//        CUSPARSE_OPERATION_NON_TRANSPOSE,
//        &alpha,
//        Q,
//        vecX,
//        &beta,
//        vecY,
//        CUDA_R_32F,
//        CUSPARSE_SPMV_ALG_DEFAULT,
//        &bufferSize);
//    cudaMalloc(&dBuffer, bufferSize);
//    // Perform Q * x
//    cusparseSpMV(handle,
//        CUSPARSE_OPERATION_NON_TRANSPOSE,
//        &alpha,
//        Q,
//        vecX,
//        &beta,
//        vecY,
//        CUDA_R_32F,
//        CUSPARSE_SPMV_ALG_DEFAULT,
//        dBuffer);
//
//    float* Qx_ptr;
//    cusparseDnVecGetValues(vecY, (void**)&Qx_ptr);
//    
//    // Compute the dot product x^T * (Q * x) using cuBLAS
//    float result;
//    cublasSdot(cublasHandle, n, in_x, 1, Qx_ptr, 1, &result);
//    // Clean up
//    cusparseDestroyDnVec(vecX);
//    cusparseDestroyDnVec(vecY);
//    cudaFree(dBuffer);
//    cudaFree(Qx);
//    cudaFree(in_x);
//    return result;
//}


float qubo_eng(cublasHandle_t cublasHandle,
    cusparseHandle_t handle,
    const cusparseSpMatDescr_t& Q,
    float* sol_vector) {
    int64_t rows, cols, nnz;
    int* d_csrOffsets, * d_cols;
    float* d_vals;
    int* dA_csrOffsets, * dA_columns;
    float* dA_values;

    cusparseIndexType_t csrRowOffsetsType;
    cusparseIndexType_t csrColIndType;
    cusparseIndexBase_t idxBase;
    cudaDataType valueTyp;

    CHECK_CUSPARSE(cusparseSpMatGetSize(Q, &rows, &cols, &nnz));

    CHECK_CUDA(cudaMalloc((void**)&d_csrOffsets, (rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_cols, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_vals, nnz * sizeof(float)));

    CHECK_CUSPARSE(cusparseCsrGet(Q, &rows, &cols, &nnz,
        (void**)&d_csrOffsets, (void**)&d_cols, (void**)&d_vals,
        &csrRowOffsetsType, &csrColIndType, &idxBase, &valueTyp));
    // Host problem definition
    const int A_num_rows = rows;
    const int A_num_cols = cols;
    const int A_nnz = nnz;
    //int h_test[11];
    //int* h_test = (int*)calloc(rows+1, sizeof(int));
    float     alpha = 1.0f;
    float     beta = 0.0f;
    //CHECK_CUDA(cudaMemcpy(h_test, d_csrOffsets, (rows + 1) * sizeof(int), cudaMemcpyDeviceToHost));
    ////cudaMemcpy(h_test, d_csrOffsets, (A_num_rows + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    ////CHECK_CUDA(cudaMemcpy(hCsrRowOffsets_toverify, dCsrRowOffsets_toverify, (rows + 1) * sizeof(int), cudaMemcpyDeviceToHost));
    //for (int i = 0; i < A_num_rows+1; i++) {
    //    std::cout << h_test[i] << std::endl;
    //}
    //--------------------------------------------------------------------------
    // Device memory management
    float * dX, * dY;
    CHECK_CUDA(cudaMalloc((void**)&dA_csrOffsets, (A_num_rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&dA_columns, A_nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&dA_values, A_nnz * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&dX, A_num_cols * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&dY, A_num_rows * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(dA_csrOffsets, d_csrOffsets, (A_num_rows + 1) * sizeof(int), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(dA_columns, d_cols, A_nnz * sizeof(int), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(dA_values, d_vals, A_nnz * sizeof(float), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(dX, sol_vector, A_num_cols * sizeof(float), cudaMemcpyDeviceToDevice));
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    void* dBuffer = NULL;
    size_t               bufferSize = 0;
    CHECK_CUSPARSE(cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
        dA_csrOffsets, dA_columns, dA_values,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    // Create dense vector X
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, A_num_cols, dX, CUDA_R_32F));
        // Create dense vector y
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, A_num_rows, dY, CUDA_R_32F));
        // allocate an external buffer if needed
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
        CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

        // execute SpMV
    CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
        CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));

        // destroy matrix/vector descriptors
    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecX));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecY));

    /*CHECK_CUDA(cudaMemcpy(hY, dY, A_num_rows * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "Result of Q*x: " << std::endl;
    for (int i = 0; i < A_num_rows; i++) {
        std::cout << hY[i] << std::endl;
    }*/

    float result;
    CHECK_CUBLAS(cublasSdot(cublasHandle, A_num_rows, dX, 1, dY, 1, &result));
    //std::cout << "Energy: " << result << std::endl;
    return result;
}

float calculate_qubo_energy(cublasHandle_t cublasHandle,
    cusparseHandle_t cusparseHandle,
    int n,
    const cusparseSpMatDescr_t& Q,
    thrust::device_vector<float>& x) {
    float alpha = 1.0f;
    float beta = 0.0f;
    size_t bufferSize = 0;
    void* dBuffer = nullptr;

    // Create dense vector descriptors
    cusparseDnVecDescr_t vecX, vecY;
    float* Qx;
    float* Qx_ptr;
    //float* hY = (float*)calloc(n, sizeof(float));
    float* in_x;

   /* for (int i = 0; i < n; i++) {
        hY[i] = 0.0f;
    }*/

    CHECK_CUDA(cudaMalloc((void**)&Qx, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&Qx_ptr, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&in_x, n * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(in_x, thrust::raw_pointer_cast(x.data()), n * sizeof(float), cudaMemcpyDeviceToDevice));
    //CHECK_CUDA(cudaMemcpy(Qx, hY, n * sizeof(float), cudaMemcpyHostToDevice));

    CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, n, in_x, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, n, Qx, CUDA_R_32F));

    // Allocate buffer
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(cusparseHandle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        Q,
        vecX,
        &beta,
        vecY,
        CUDA_R_32F,
        CUSPARSE_SPMV_ALG_DEFAULT,
        &bufferSize));
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

    // Perform Q * x
    CHECK_CUSPARSE(cusparseSpMV(cusparseHandle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        Q,
        vecX,
        &beta,
        vecY,
        CUDA_R_32F,
        CUSPARSE_SPMV_ALG_DEFAULT,
        dBuffer));

    // Extract the raw pointers to the data in the dense vector descriptors
    //float* Qx_ptr;
    //CHECK_CUSPARSE(cusparseDnVecGetValues(vecY, (void**)&Qx_ptr));
    // Ensure the computation is complete before accessing the result
    CHECK_CUDA(cudaDeviceSynchronize());

    /*CHECK_CUDA(cudaMemcpy(hY, Qx, n * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; i++) {
        std::cout << hY[i] << std::endl;
    }*/

    // Compute the dot product x^T * (Q * x) using cuBLAS
    float result;
    CHECK_CUBLAS(cublasSdot(cublasHandle, n, in_x, 1, Qx, 1, &result));

    // Clean up
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecX));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecY));
    CHECK_CUDA(cudaFree(dBuffer));
    CHECK_CUDA(cudaFree(Qx));
    CHECK_CUDA(cudaFree(in_x));

    return result;
}

//void brute_force_solutions(cublasHandle_t cublasHandle,
//    cusparseHandle_t handle,
//    int n,
//    const cusparseSpMatDescr_t& Q) {
//
//    int num_solutions = 1 << n;  // 2^n
//    thrust::device_vector<float> d_solutions(num_solutions * n);
//    for (int i = 0; i < num_solutions; ++i) {
//        for (int j = 0; j < n; ++j) {
//            float val = static_cast<float>((i & (1 << j)) != 0);
//            d_solutions[i * n + j] = val;
//        }
//    }
//
//
//    // Allocate memory for energies
//    thrust::device_vector<float> d_energies(num_solutions);
//    thrust::device_vector<float> sol_x(n);
//    for (int i = 0; i < num_solutions; ++i) {
//        thrust::copy(d_solutions.begin() + (i * n), d_solutions.begin() + ((i + 1) * n), sol_x.begin());
//        float eng = calculate_qubo_energy(cublasHandle, handle, n, Q, sol_x);
//        d_energies[i] = eng;
//        std::cout << "Energy i: " << eng << std::endl;
//    }
//
//
//    // Find the minimum energy
//    auto min_energy_iter = thrust::min_element(d_energies.begin(), d_energies.end());
//    float min_energy = *min_energy_iter;
//    int min_index = min_energy_iter - d_energies.begin();
//
//    // Copy the best solution to host
//    thrust::host_vector<float> h_best_solution(n);
//    thrust::copy(d_solutions.begin() + min_index * n, d_solutions.begin() + (min_index + 1) * n, h_best_solution.begin());
//
//    // Print the result
//    std::cout << "Best energy: " << min_energy << std::endl;
//    std::cout << "Best solution: ";
//    for (float bit : h_best_solution) {
//        std::cout << bit << " ";
//    }
//    std::cout << std::endl;
//    convert_sparse_to_dense_and_display(handle, Q, n);
//}

void brute_force_solutions(cublasHandle_t cublasHandle,
    cusparseHandle_t handle,
    int n,
    const cusparseSpMatDescr_t& Q) {
    indicators::show_console_cursor(false);
    int num_solutions = 1 << n;  // 2^n
    thrust::device_vector<float> d_solutions(num_solutions * n);
    /*
    for (int i = 0; i < num_solutions; ++i) {
        for (int j = 0; j < n; ++j) {
            float val = static_cast<float>((i & (1 << j)) != 0);
            d_solutions[i * n + j] = val;
        }
    }*/

    //convert_sparse_to_dense_and_display(handle, Q, n);
    // Allocate memory for energies
    thrust::device_vector<float> d_energies(num_solutions);
    thrust::device_vector<float> sol_x(n);
    
    indicators::ProgressBar  bar{
    indicators::option::BarWidth{40},
    indicators::option::Start{"["},
    indicators::option::Fill{"="},
    indicators::option::Lead{">"},
    indicators::option::Remainder{" "},
    indicators::option::End{" ]"},
    indicators::option::ForegroundColor{indicators::Color::white},
    indicators::option::FontStyles{
          std::vector<indicators::FontStyle>{indicators::FontStyle::bold}},
    indicators::option::MaxProgress{num_solutions}
    };
    int smallest_ind = -1;
    float smallest_eng = 1000;

    for (int i = 0; i < num_solutions; ++i) {
        for (int j = 0; j < n; ++j) {
            float val = static_cast<float>((i & (1 << j)) != 0);
            sol_x[j] = val;
            d_solutions[i * n + j] = val;
        }
        float eng = qubo_eng(cublasHandle, handle, Q, thrust::raw_pointer_cast(sol_x.data()));
        //float eng = calculate_qubo_energy(cublasHandle, handle, n, Q, sol_x);
        d_energies[i] = eng;
        if (eng <= smallest_eng) {
            smallest_ind = i;
            smallest_eng = eng;
        }
        //std::cout << "Energy i: " << eng << std::endl;
        bar.set_option(indicators::option::PostfixText{ std::to_string(i) + "/" + std::to_string(num_solutions) });
        bar.tick();
    }
    std::cout << "Tested all solutions, now sorting." << std::endl;
    // Find the minimum energy
    auto min_energy_iter = thrust::min_element(d_energies.begin(), d_energies.end());
    float min_energy = *min_energy_iter;
    int min_index = min_energy_iter - d_energies.begin();

    // Copy the best solution to host
    thrust::host_vector<float> h_best_solution(n);
    thrust::copy(d_solutions.begin() + smallest_ind * n, d_solutions.begin() + (smallest_ind + 1) * n, h_best_solution.begin());

    // Print the result
    std::cout << "Best energy my: " << smallest_eng << " and id: " << smallest_ind << std::endl;
    std::cout << "Best energy: " << min_energy << " and id: " << min_index << std::endl;
    std::cout << "Best solution: ";
    for (float bit : h_best_solution) {
        std::cout << bit << " ";
    }
    std::cout << std::endl;
    convert_sparse_to_dense_and_display(handle, Q, n);
}


int main() {
    int n = 12;
    int seed = 14;
    float density = 0.5;
    std::mt19937 rng(seed);

    std::vector<int> p = generate_initial_permutation(rng, n);

    int split = estimate_split(density, n);  // Example split, can be computed as needed
    std::cout << "Split: " << split << std::endl;

    thrust::device_vector<int> d_rows;
    thrust::device_vector<int> d_cols;
    thrust::device_vector<float> d_vals;

    create_graph_sparse(n, split, p, d_rows, d_cols, d_vals);
    thrust::device_vector<float> d_x = generate_solution(p, split);
    std::cout << "Graph created.";
    // Print the result (for debugging purposes)
    thrust::host_vector<int> h_rows = d_rows;
    thrust::host_vector<int> h_cols = d_cols;
    thrust::host_vector<float> h_vals = d_vals;
    thrust::host_vector<float> h_x = d_x;
    int print_size = 2*n;
    std::cout << "Rows: ";
    for (int i = 0; i < print_size; ++i) {
        std::cout << h_rows[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Cols: ";
    for (int i = 0; i < print_size; ++i) {
        std::cout << h_cols[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Vals: ";
    for (int i = 0; i < print_size; ++i) {
        std::cout << h_vals[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Solution: ";
    for (int i = 0; i < n; ++i) {
        std::cout << h_x[i] << " ";
    }
    std::cout << std::endl;

    cusparseHandle_t handle;
    cublasHandle_t cublasHandle;
    cusparseSpMatDescr_t graph_csr, Q;

    // Initialize cuSPARSE
    cublasCreate(&cublasHandle);
    CHECK_CUSPARSE(cusparseCreate(&handle));
    int nnz = d_vals.size();
    thrust::device_vector<int> d_csrOffsets(n + 1);
    cusparseXcoo2csr(handle,
        thrust::raw_pointer_cast(d_rows.data()),
        nnz,
        n,
        thrust::raw_pointer_cast(d_csrOffsets.data()),
        CUSPARSE_INDEX_BASE_ZERO);

    CHECK_CUSPARSE(cusparseCreateCsr(&graph_csr,
        n,
        n,
        nnz,
        thrust::raw_pointer_cast(d_csrOffsets.data()),
        thrust::raw_pointer_cast(d_cols.data()), // column indices
        thrust::raw_pointer_cast(d_vals.data()), // values
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    thrust::device_vector<float> d_QVals(nnz);
    CHECK_CUSPARSE(cusparseCreateCsr(&Q,
        n,
        n,
        nnz,
        thrust::raw_pointer_cast(d_csrOffsets.data()),
        thrust::raw_pointer_cast(d_cols.data()), // column indices
        thrust::raw_pointer_cast(d_QVals.data()), // values
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    // Now matDescr can be used in further cuSPARSE operations

    int* newRowsPtr;
    int* newCols;
    float* newVals;
    csr_data extended_sparse_data;

    CHECK_CUDA(cudaMalloc((void**)&newRowsPtr, (n + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&newCols, (nnz + n) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&newVals, (nnz + n) * sizeof(float)));

    extended_sparse_data.rowPointer = newRowsPtr;
    extended_sparse_data.cols = newCols;
    extended_sparse_data.vals = newVals;


    std::cout << "Sparse matrix created successfully!" << std::endl;
    graph_to_qubo(handle, graph_csr, Q, extended_sparse_data);
    /*convert_sparse_to_dense_and_display(handle, graph_csr, n); */
    //convert_sparse_to_dense_and_display(handle, Q, n);
    
    float planted_energy = qubo_eng(cublasHandle, handle, Q, thrust::raw_pointer_cast(d_x.data()));
    
    std::cout << "Qubo energy of planted solution: " << planted_energy << std::endl;

    /*float energy = calculate_qubo_energy(cublasHandle, handle, n, Q, d_x);
    std::cout << "Qubo energy: " << energy << std::endl;*/
    brute_force_solutions(cublasHandle, handle, n, Q);
    
    // Clean up
    CHECK_CUSPARSE(cusparseDestroySpMat(graph_csr));
    CHECK_CUSPARSE(cusparseDestroySpMat(Q));
    CHECK_CUSPARSE(cusparseDestroy(handle));

    return 0;
};
