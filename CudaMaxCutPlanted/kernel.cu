#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/random.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <iostream>
#include <algorithm>
#include <random>
#include <vector>
#include <iostream>
#include <iomanip> // For std::fixed and std::setprecision


#define CUDA_CALL(x) do { if((x) != cudaSuccess) { printf("Error at %s:%d\n",__FILE__,__LINE__); return EXIT_FAILURE;}} while(0)
#define CUSPARSE_CALL(x) do { if((x) != CUSPARSE_STATUS_SUCCESS) { printf("Error at %s:%d\n",__FILE__,__LINE__); return EXIT_FAILURE;}} while(0)

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

void convert_sparse_to_dense_and_display(cusparseHandle_t handle, cusparseSpMatDescr_t& matDescr, int n) {
    // Allocate memory for the dense matrix on the device
    float* d_denseMat;
    cudaMalloc((void**)&d_denseMat, n * n * sizeof(float));

    // Create a dense matrix descriptor
    cusparseDnMatDescr_t denseDescr;
    cusparseCreateDnMat(&denseDescr,
        n, // number of rows
        n, // number of columns
        n, // leading dimension
        d_denseMat, // pointer to dense matrix data
        CUDA_R_32F, // data type
        CUSPARSE_ORDER_ROW); // row-major order

    // Convert sparse matrix to dense matrix
    void* dBuffer = NULL;
    size_t bufferSize = 0;
    cusparseSparseToDense_bufferSize(handle,
        matDescr,
        denseDescr,
        CUSPARSE_SPARSETODENSE_ALG_DEFAULT,
        &bufferSize);

    cudaMalloc(&dBuffer, bufferSize);

    cusparseSparseToDense(handle,
        matDescr,
        denseDescr,
        CUSPARSE_SPARSETODENSE_ALG_DEFAULT,
        dBuffer);

    // Copy the dense matrix from device to host
    std::vector<float> h_denseMat(n * n);
    cudaMemcpy(h_denseMat.data(), d_denseMat, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << std::fixed << std::setprecision(4); // Set precision to 2 decimal places
    std::cout << "Dense matrix:" << std::endl;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << h_denseMat[i * n + j] << " ";
        }
        std::cout << std::endl;
    }

    // Clean up
    cudaFree(d_denseMat);
    cudaFree(dBuffer);
    cusparseDestroyDnMat(denseDescr);
}


void fill_diagonal(cusparseHandle_t handle, cusparseSpMatDescr_t& input, thrust::device_vector<float> diag) {
    int64_t n, nnz;
    int* d_csrOffsets;
    int* d_cols;
    float* d_vals;

    cusparseIndexType_t csrRowOffsetsType;
    cusparseIndexType_t csrColIndType;
    cusparseIndexBase_t idxBase;
    cudaDataType valueTyp;

    cusparseCsrGet(input, &n, &n, &nnz, (void**)&d_csrOffsets, (void**)&d_cols, (void**)&d_vals, &csrRowOffsetsType, &csrColIndType, &idxBase, &valueTyp);

    thrust::device_vector<int> d_rows_tdv(nnz);
    thrust::device_vector<int> d_cols_tdv(d_cols, d_cols+nnz);
    thrust::device_vector<float> d_vals_tdv(d_vals, d_vals+nnz);

    cusparseXcsr2coo(handle,
        d_csrOffsets,
        nnz,
        n,
        thrust::raw_pointer_cast(d_rows_tdv.data()),
        CUSPARSE_INDEX_BASE_ZERO);

    d_rows_tdv.resize(nnz + n);
    d_cols_tdv.resize(nnz + n);
    d_vals_tdv.resize(nnz + n);

    thrust::device_vector<int> d_vec(n);

    // Fill the vector with values from 0 to n-1
    thrust::sequence(d_vec.begin(), d_vec.end());

    thrust::copy(d_vec.begin(), d_vec.end(), d_rows_tdv.begin() + nnz);
    thrust::copy(d_vec.begin(), d_vec.end(), d_cols_tdv.begin() + nnz);
    thrust::copy(diag.begin(), diag.end(), d_vals_tdv.begin() + nnz);

    cusparseXcsr2coo(handle,
        d_csrOffsets,
        nnz,
        n,
        thrust::raw_pointer_cast(d_rows_tdv.data()),
        CUSPARSE_INDEX_BASE_ZERO);
    thrust::device_vector<int> d_csrOffsets_o(n + 1);

    thrust::sort_by_key(thrust::make_zip_iterator(thrust::make_tuple(d_rows_tdv.begin(), d_cols_tdv.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(d_rows_tdv.end(), d_cols_tdv.end())),
        d_vals_tdv.begin());
    
    cusparseXcoo2csr(handle,
        thrust::raw_pointer_cast(d_rows_tdv.data()),
        nnz + n,
        n,
        thrust::raw_pointer_cast(d_csrOffsets_o.data()),
        CUSPARSE_INDEX_BASE_ZERO);

    cusparseCsrSetPointers(input,
        thrust::raw_pointer_cast(d_csrOffsets_o.data()),
        thrust::raw_pointer_cast(d_cols_tdv.data()),
        thrust::raw_pointer_cast(d_vals_tdv.data()));

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

void graph_to_qubo(cusparseHandle_t handle, cusparseSpMatDescr_t& graph, cusparseSpMatDescr_t& Q) {
    scale_csr_matrix(handle, -1.0f, graph, Q);
    thrust::device_vector<float> sum = sum_rows_csr_matrix(handle, graph);
    fill_diagonal(handle, Q, sum);
    scale_csr_matrix(handle, -0.25f, Q, Q);

    for (int i = 0; i < sum.size(); i++) {
        std::cout << sum[i] << std::endl;
    }
}


int main() {
    int n = 10;
    int seed = 7;
    float density = 0.5;
    std::mt19937 rng(seed);

    std::vector<int> p = generate_initial_permutation(rng, n);

    int split = estimate_split(density, n);  // Example split, can be computed as needed
    std::cout << "Split: " << split << std::endl;

    thrust::device_vector<int> d_rows;
    thrust::device_vector<int> d_cols;
    thrust::device_vector<float> d_vals;

    create_graph_sparse(n, split, p, d_rows, d_cols, d_vals);

    // Print the result (for debugging purposes)
    thrust::host_vector<int> h_rows = d_rows;
    thrust::host_vector<int> h_cols = d_cols;
    thrust::host_vector<float> h_vals = d_vals;
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

    cusparseHandle_t handle;
    cusparseSpMatDescr_t graph_csr, Q;

    // Initialize cuSPARSE
    CUSPARSE_CALL(cusparseCreate(&handle));
    int nnz = d_vals.size();
    thrust::device_vector<int> d_csrOffsets(n + 1);
    cusparseXcoo2csr(handle,
        thrust::raw_pointer_cast(d_rows.data()),
        nnz,
        n,
        thrust::raw_pointer_cast(d_csrOffsets.data()),
        CUSPARSE_INDEX_BASE_ZERO);

    // Create a sparse matrix in COO format
    //CUSPARSE_CALL(cusparseCreateCoo(&graph_coo,
    //    n, // number of rows
    //    n, // number of columns
    //    d_vals.size(), // number of non-zero elements
    //    thrust::raw_pointer_cast(d_rows.data()), // row indices
    //    thrust::raw_pointer_cast(d_cols.data()), // column indices
    //    thrust::raw_pointer_cast(d_vals.data()), // values
    //    CUSPARSE_INDEX_32I,
    //    CUSPARSE_INDEX_BASE_ZERO,
    //    CUDA_R_32F)); // data type

    CUSPARSE_CALL(cusparseCreateCsr(&graph_csr,
        n,
        n,
        nnz,
        thrust::raw_pointer_cast(d_csrOffsets.data()),
        thrust::raw_pointer_cast(d_cols.data()), // column indices
        thrust::raw_pointer_cast(d_vals.data()), // values
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    thrust::device_vector<float> d_QVals(nnz);
    CUSPARSE_CALL(cusparseCreateCsr(&Q,
        n,
        n,
        nnz,
        thrust::raw_pointer_cast(d_csrOffsets.data()),
        thrust::raw_pointer_cast(d_cols.data()), // column indices
        thrust::raw_pointer_cast(d_QVals.data()), // values
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    // Now matDescr can be used in further cuSPARSE operations
    std::cout << "Sparse matrix created successfully!" << std::endl;
    graph_to_qubo(handle, graph_csr, Q);
    convert_sparse_to_dense_and_display(handle, graph_csr, n);
    convert_sparse_to_dense_and_display(handle, Q, n);
    // Clean up
    CUSPARSE_CALL(cusparseDestroySpMat(graph_csr));
    CUSPARSE_CALL(cusparseDestroy(handle));

    return 0;
};
