//#include <cuda_runtime.h>
//#include <cusparse_v2.h>
#include "CudaSparseMatrix.hpp"
//#include <cublas_v2.h>
#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>
//#include <thrust/device_vector.h>
//#include <thrust/execution_policy.h>
#include <thrust/sort.h>
//#include <thrust/random.h>
//#include <thrust/reduce.h>
//#include <thrust/extrema.h>
//#include <thrust/sequence.h>
//#include <thrust/inner_product.h>
//#include <iostream>
//#include <algorithm>
#include <random>
#include <curand_kernel.h>
//#include <vector>
//#include <iostream>
//#include <iomanip>
#include "indicators.hpp"
#ifdef __INTELLISENSE__
#include "intellisense_cuda_intrinsics.h"
#endif



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
    CHECK_CUDA(cudaMalloc((void**)&states, nnz * sizeof(states)));
    int gridSize = (nnz + BLOCK_SIZE - 1) / BLOCK_SIZE;

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

    //thrust::host_vector<int> h_rows, h_cols;
    //thrust::host_vector<float> h_vals;

    //thrust::default_random_engine rng;
    //thrust::random::uniform_real_distribution<float> dist(0.01f, 1.0f);

    //std::vector<std::tuple<int, int, float>> combined;

    //for (int i = 0; i < split; ++i) {
    //    std::cout << "P_" << i << ":" << p[i] << std::endl;
    //    for (int j = split; j < n; ++j) {
    //        std::cout << "I" << ": " << i << " ";
    //        std::cout << "J" << ":" << j << std::endl;
    //        float rnd_val = dist(rng);
    //        combined.push_back(std::make_tuple(p[i], p[j], rnd_val));
    //        //combined.push_back(std::make_tuple(p[j], p[i], rnd_val));
    //    }
    //}

    //std::sort(combined.begin(), combined.end(), [](const auto& a, const auto& b) {
    //    if (std::get<0>(a) == std::get<0>(b)) {
    //        return std::get<1>(a) < std::get<1>(b);
    //    }
    //    return std::get<0>(a) < std::get<0>(b);
    //    });

    //for (size_t i = 0; i < combined.size(); ++i) {
    //    
    //    
    //    h_rows.push_back(std::get<0>(combined[i]));
    //    h_cols.push_back(std::get<1>(combined[i]));
    //    h_vals.push_back(std::get<2>(combined[i]));
    //}

    //d_rows = h_rows;
    //d_cols = h_cols;
    //d_vals = h_vals;
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
    Q.multiply(-1.0f);
    float* row_sum = Q.sum(0);    
    Q.fill_diagonal(row_sum);
    /*scale_csr_matrix(handle, -1.0f, graph, Q);
    thrust::device_vector<float> sum = sum_rows_csr_matrix(handle, graph);
    fill_diagonal(handle, Q, sum, extended_pointers);
    scale_csr_matrix(handle, -0.25f, Q, Q);

    for (int i = 0; i < sum.size(); i++) {
        std::cout << sum[i] << std::endl;
    }*/
}

//
//float qubo_eng(cublasHandle_t cublasHandle,
//    cusparseHandle_t handle,
//    const cusparseSpMatDescr_t& Q,
//    float* sol_vector) {
//    int64_t rows, cols, nnz;
//    int* d_csrOffsets, * d_cols;
//    float* d_vals;
//    int* dA_csrOffsets, * dA_columns;
//    float* dA_values;
//
//    cusparseIndexType_t csrRowOffsetsType;
//    cusparseIndexType_t csrColIndType;
//    cusparseIndexBase_t idxBase;
//    cudaDataType valueTyp;
//
//    CHECK_CUSPARSE(cusparseSpMatGetSize(Q, &rows, &cols, &nnz));
//
//    CHECK_CUDA(cudaMalloc((void**)&d_csrOffsets, (rows + 1) * sizeof(int)));
//    CHECK_CUDA(cudaMalloc((void**)&d_cols, nnz * sizeof(int)));
//    CHECK_CUDA(cudaMalloc((void**)&d_vals, nnz * sizeof(float)));
//
//    CHECK_CUSPARSE(cusparseCsrGet(Q, &rows, &cols, &nnz,
//        (void**)&d_csrOffsets, (void**)&d_cols, (void**)&d_vals,
//        &csrRowOffsetsType, &csrColIndType, &idxBase, &valueTyp));
//    // Host problem definition
//    const int A_num_rows = rows;
//    const int A_num_cols = cols;
//    const int A_nnz = nnz;
//    //int h_test[11];
//    //int* h_test = (int*)calloc(rows+1, sizeof(int));
//    float     alpha = 1.0f;
//    float     beta = 0.0f;
//    //CHECK_CUDA(cudaMemcpy(h_test, d_csrOffsets, (rows + 1) * sizeof(int), cudaMemcpyDeviceToHost));
//    ////cudaMemcpy(h_test, d_csrOffsets, (A_num_rows + 1) * sizeof(int), cudaMemcpyDeviceToHost);
//    ////CHECK_CUDA(cudaMemcpy(hCsrRowOffsets_toverify, dCsrRowOffsets_toverify, (rows + 1) * sizeof(int), cudaMemcpyDeviceToHost));
//    //for (int i = 0; i < A_num_rows+1; i++) {
//    //    std::cout << h_test[i] << std::endl;
//    //}
//    //--------------------------------------------------------------------------
//    // Device memory management
//    float* dX, * dY;
//    CHECK_CUDA(cudaMalloc((void**)&dA_csrOffsets, (A_num_rows + 1) * sizeof(int)));
//    CHECK_CUDA(cudaMalloc((void**)&dA_columns, A_nnz * sizeof(int)));
//    CHECK_CUDA(cudaMalloc((void**)&dA_values, A_nnz * sizeof(float)));
//    CHECK_CUDA(cudaMalloc((void**)&dX, A_num_cols * sizeof(float)));
//    CHECK_CUDA(cudaMalloc((void**)&dY, A_num_rows * sizeof(float)));
//
//    CHECK_CUDA(cudaMemcpy(dA_csrOffsets, d_csrOffsets, (A_num_rows + 1) * sizeof(int), cudaMemcpyDeviceToDevice));
//    CHECK_CUDA(cudaMemcpy(dA_columns, d_cols, A_nnz * sizeof(int), cudaMemcpyDeviceToDevice));
//    CHECK_CUDA(cudaMemcpy(dA_values, d_vals, A_nnz * sizeof(float), cudaMemcpyDeviceToDevice));
//    CHECK_CUDA(cudaMemcpy(dX, sol_vector, A_num_cols * sizeof(float), cudaMemcpyDeviceToDevice));
//    cusparseSpMatDescr_t matA;
//    cusparseDnVecDescr_t vecX, vecY;
//    void* dBuffer = NULL;
//    size_t               bufferSize = 0;
//    CHECK_CUSPARSE(cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
//        dA_csrOffsets, dA_columns, dA_values,
//        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
//        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
//    // Create dense vector X
//    CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, A_num_cols, dX, CUDA_R_32F));
//    // Create dense vector y
//    CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, A_num_rows, dY, CUDA_R_32F));
//    // allocate an external buffer if needed
//    CHECK_CUSPARSE(cusparseSpMV_bufferSize(
//        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
//        &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
//        CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
//    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));
//
//    // execute SpMV
//    CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
//        &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
//        CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
//
//    // destroy matrix/vector descriptors
//    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
//    CHECK_CUSPARSE(cusparseDestroyDnVec(vecX));
//    CHECK_CUSPARSE(cusparseDestroyDnVec(vecY));
//
//    /*CHECK_CUDA(cudaMemcpy(hY, dY, A_num_rows * sizeof(float), cudaMemcpyDeviceToHost));
//    std::cout << "Result of Q*x: " << std::endl;
//    for (int i = 0; i < A_num_rows; i++) {
//        std::cout << hY[i] << std::endl;
//    }*/
//
//    float result;
//    CHECK_CUBLAS(cublasSdot(cublasHandle, A_num_rows, dX, 1, dY, 1, &result));
//    //std::cout << "Energy: " << result << std::endl;
//    return result;
//}

//float calculate_qubo_energy(cublasHandle_t cublasHandle,
//    cusparseHandle_t cusparseHandle,
//    int n,
//    const cusparseSpMatDescr_t& Q,
//    thrust::device_vector<float>& x) {
//    float alpha = 1.0f;
//    float beta = 0.0f;
//    size_t bufferSize = 0;
//    void* dBuffer = nullptr;
//
//    // Create dense vector descriptors
//    cusparseDnVecDescr_t vecX, vecY;
//    float* Qx;
//    float* Qx_ptr;
//    //float* hY = (float*)calloc(n, sizeof(float));
//    float* in_x;
//
//    /* for (int i = 0; i < n; i++) {
//         hY[i] = 0.0f;
//     }*/
//
//    CHECK_CUDA(cudaMalloc((void**)&Qx, n * sizeof(float)));
//    CHECK_CUDA(cudaMalloc((void**)&Qx_ptr, n * sizeof(float)));
//    CHECK_CUDA(cudaMalloc((void**)&in_x, n * sizeof(float)));
//
//    CHECK_CUDA(cudaMemcpy(in_x, thrust::raw_pointer_cast(x.data()), n * sizeof(float), cudaMemcpyDeviceToDevice));
//    //CHECK_CUDA(cudaMemcpy(Qx, hY, n * sizeof(float), cudaMemcpyHostToDevice));
//
//    CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, n, in_x, CUDA_R_32F));
//    CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, n, Qx, CUDA_R_32F));
//
//    // Allocate buffer
//    CHECK_CUSPARSE(cusparseSpMV_bufferSize(cusparseHandle,
//        CUSPARSE_OPERATION_NON_TRANSPOSE,
//        &alpha,
//        Q,
//        vecX,
//        &beta,
//        vecY,
//        CUDA_R_32F,
//        CUSPARSE_SPMV_ALG_DEFAULT,
//        &bufferSize));
//    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));
//
//    // Perform Q * x
//    CHECK_CUSPARSE(cusparseSpMV(cusparseHandle,
//        CUSPARSE_OPERATION_NON_TRANSPOSE,
//        &alpha,
//        Q,
//        vecX,
//        &beta,
//        vecY,
//        CUDA_R_32F,
//        CUSPARSE_SPMV_ALG_DEFAULT,
//        dBuffer));
//
//    // Extract the raw pointers to the data in the dense vector descriptors
//    //float* Qx_ptr;
//    //CHECK_CUSPARSE(cusparseDnVecGetValues(vecY, (void**)&Qx_ptr));
//    // Ensure the computation is complete before accessing the result
//    CHECK_CUDA(cudaDeviceSynchronize());
//
//    /*CHECK_CUDA(cudaMemcpy(hY, Qx, n * sizeof(float), cudaMemcpyDeviceToHost));
//    for (int i = 0; i < n; i++) {
//        std::cout << hY[i] << std::endl;
//    }*/
//
//    // Compute the dot product x^T * (Q * x) using cuBLAS
//    float result;
//    CHECK_CUBLAS(cublasSdot(cublasHandle, n, in_x, 1, Qx, 1, &result));
//
//    // Clean up
//    CHECK_CUSPARSE(cusparseDestroyDnVec(vecX));
//    CHECK_CUSPARSE(cusparseDestroyDnVec(vecY));
//    CHECK_CUDA(cudaFree(dBuffer));
//    CHECK_CUDA(cudaFree(Qx));
//    CHECK_CUDA(cudaFree(in_x));
//
//    return result;
//}
//
//void brute_force_solutions(cublasHandle_t cublasHandle,
//    cusparseHandle_t handle,
//    int n,
//    const cusparseSpMatDescr_t& Q) {
//    indicators::show_console_cursor(false);
//    int num_solutions = 1 << n;  // 2^n
//    thrust::device_vector<float> d_solutions(num_solutions * n);
//    /*
//    for (int i = 0; i < num_solutions; ++i) {
//        for (int j = 0; j < n; ++j) {
//            float val = static_cast<float>((i & (1 << j)) != 0);
//            d_solutions[i * n + j] = val;
//        }
//    }*/
//
//    //convert_sparse_to_dense_and_display(handle, Q, n);
//    // Allocate memory for energies
//    thrust::device_vector<float> d_energies(num_solutions);
//    thrust::device_vector<float> sol_x(n);
//
//    indicators::ProgressBar  bar{
//    indicators::option::BarWidth{40},
//    indicators::option::Start{"["},
//    indicators::option::Fill{"="},
//    indicators::option::Lead{">"},
//    indicators::option::Remainder{" "},
//    indicators::option::End{" ]"},
//    indicators::option::ForegroundColor{indicators::Color::white},
//    indicators::option::FontStyles{
//          std::vector<indicators::FontStyle>{indicators::FontStyle::bold}},
//    indicators::option::MaxProgress{num_solutions}
//    };
//    int smallest_ind = -1;
//    float smallest_eng = 1000;
//
//    for (int i = 0; i < num_solutions; ++i) {
//        for (int j = 0; j < n; ++j) {
//            float val = static_cast<float>((i & (1 << j)) != 0);
//            sol_x[j] = val;
//            d_solutions[i * n + j] = val;
//        }
//        float eng = qubo_eng(cublasHandle, handle, Q, thrust::raw_pointer_cast(sol_x.data()));
//        //float eng = calculate_qubo_energy(cublasHandle, handle, n, Q, sol_x);
//        d_energies[i] = eng;
//        if (eng <= smallest_eng) {
//            smallest_ind = i;
//            smallest_eng = eng;
//        }
//        //std::cout << "Energy i: " << eng << std::endl;
//        bar.set_option(indicators::option::PostfixText{ std::to_string(i) + "/" + std::to_string(num_solutions) });
//        bar.tick();
//    }
//    std::cout << "Tested all solutions, now sorting." << std::endl;
//    // Find the minimum energy
//    auto min_energy_iter = thrust::min_element(d_energies.begin(), d_energies.end());
//    float min_energy = *min_energy_iter;
//    int min_index = min_energy_iter - d_energies.begin();
//
//    // Copy the best solution to host
//    thrust::host_vector<float> h_best_solution(n);
//    thrust::copy(d_solutions.begin() + smallest_ind * n, d_solutions.begin() + (smallest_ind + 1) * n, h_best_solution.begin());
//
//    // Print the result
//    std::cout << "Best energy my: " << smallest_eng << " and id: " << smallest_ind << std::endl;
//    std::cout << "Best energy: " << min_energy << " and id: " << min_index << std::endl;
//    std::cout << "Best solution: ";
//    for (float bit : h_best_solution) {
//        std::cout << bit << " ";
//    }
//    std::cout << std::endl;
//    convert_sparse_to_dense_and_display(handle, Q, n);
//}

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
    int n = 12;
    int seed = 14;
    float density = 0.5;
    int* I, * J;
    float* V;
    std::mt19937 rng(seed);

    int* p = generate_initial_permutation(rng, n);

    int split = estimate_split(density, n);  // Example split, can be computed as needed
    int nnz = split * (n - split );
    std::cout << "Split: " << split << " nnz: " << nnz << std::endl;

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

    cudaFree(I);
    cudaFree(J);
    cudaFree(V);
    cudaFree(p);
    cudaFree(x);

    delete[] h_x;

    //std::cout << "Graph created.";
    //// Print the result (for debugging purposes)
    //thrust::host_vector<int> h_rows = d_rows;
    //thrust::host_vector<int> h_cols = d_cols;
    //thrust::host_vector<float> h_vals = d_vals;
    //thrust::host_vector<float> h_x = d_x;
    //int print_size = 2 * n;
    //std::cout << "Rows: ";
    //for (int i = 0; i < print_size; ++i) {
    //    std::cout << h_rows[i] << " ";
    //}
    //std::cout << std::endl;

    //std::cout << "Cols: ";
    //for (int i = 0; i < print_size; ++i) {
    //    std::cout << h_cols[i] << " ";
    //}
    //std::cout << std::endl;

    //std::cout << "Vals: ";
    //for (int i = 0; i < print_size; ++i) {
    //    std::cout << h_vals[i] << " ";
    //}
    //std::cout << std::endl;

    //std::cout << "Solution: ";
    //for (int i = 0; i < n; ++i) {
    //    std::cout << h_x[i] << " ";
    //}
    //std::cout << std::endl;

    //cusparseHandle_t handle;
    //cublasHandle_t cublasHandle;
    //cusparseSpMatDescr_t graph_csr, Q;

    //// Initialize cuSPARSE
    //cublasCreate(&cublasHandle);
    //CHECK_CUSPARSE(cusparseCreate(&handle));
    //int nnz = d_vals.size();
    //thrust::device_vector<int> d_csrOffsets(n + 1);
    //cusparseXcoo2csr(handle,
    //    thrust::raw_pointer_cast(d_rows.data()),
    //    nnz,
    //    n,
    //    thrust::raw_pointer_cast(d_csrOffsets.data()),
    //    CUSPARSE_INDEX_BASE_ZERO);

    //CHECK_CUSPARSE(cusparseCreateCsr(&graph_csr,
    //    n,
    //    n,
    //    nnz,
    //    thrust::raw_pointer_cast(d_csrOffsets.data()),
    //    thrust::raw_pointer_cast(d_cols.data()), // column indices
    //    thrust::raw_pointer_cast(d_vals.data()), // values
    //    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
    //    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    //thrust::device_vector<float> d_QVals(nnz);
    //CHECK_CUSPARSE(cusparseCreateCsr(&Q,
    //    n,
    //    n,
    //    nnz,
    //    thrust::raw_pointer_cast(d_csrOffsets.data()),
    //    thrust::raw_pointer_cast(d_cols.data()), // column indices
    //    thrust::raw_pointer_cast(d_QVals.data()), // values
    //    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
    //    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    //// Now matDescr can be used in further cuSPARSE operations

    //int* newRowsPtr;
    //int* newCols;
    //float* newVals;
    //csr_data extended_sparse_data;

    //CHECK_CUDA(cudaMalloc((void**)&newRowsPtr, (n + 1) * sizeof(int)));
    //CHECK_CUDA(cudaMalloc((void**)&newCols, (nnz + n) * sizeof(int)));
    //CHECK_CUDA(cudaMalloc((void**)&newVals, (nnz + n) * sizeof(float)));

    //extended_sparse_data.rowPointer = newRowsPtr;
    //extended_sparse_data.cols = newCols;
    //extended_sparse_data.vals = newVals;


    //std::cout << "Sparse matrix created successfully!" << std::endl;
    //graph_to_qubo(handle, graph_csr, Q, extended_sparse_data);
    ///*convert_sparse_to_dense_and_display(handle, graph_csr, n); */
    ////convert_sparse_to_dense_and_display(handle, Q, n);

    //float planted_energy = qubo_eng(cublasHandle, handle, Q, thrust::raw_pointer_cast(d_x.data()));

    //std::cout << "Qubo energy of planted solution: " << planted_energy << std::endl;

    ///*float energy = calculate_qubo_energy(cublasHandle, handle, n, Q, d_x);
    //std::cout << "Qubo energy: " << energy << std::endl;*/
    //brute_force_solutions(cublasHandle, handle, n, Q);

    //// Clean up
    //CHECK_CUSPARSE(cusparseDestroySpMat(graph_csr));
    //CHECK_CUSPARSE(cusparseDestroySpMat(Q));
    //CHECK_CUSPARSE(cusparseDestroy(handle));

    return 0;
};
