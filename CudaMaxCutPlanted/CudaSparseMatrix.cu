#include "CudaSparseMatrix.hpp"

#include <iostream>
#include <iomanip>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "SparseMatrixSumKernel.cuh"

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

CudaSparseMatrix::CudaSparseMatrix(int* I, int* J, float* V, int n, int nnz, SparseType sparseType, MemoryType memType): n_(n), nnz_(nnz)
{
    cusparseHandle_t& cusparseHandle_ = CusparseHandle::getInstance();
    CHECK_CUDA(cudaMalloc((void**)&d_csrOffsets_, (n_ + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_cols_, nnz_ * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_vals_, nnz_ * sizeof(float)));

    allocateAndCopy(I, J, V, sparseType, memType);

    CHECK_CUSPARSE(cusparseCreateCsr(&matDescr_, n_, n_, nnz_,
        d_csrOffsets_, d_cols_, d_vals_,
        csr_row_ind_type_, csr_col_ind_type_,
        index_base_, valueType_));
}

CudaSparseMatrix::CudaSparseMatrix(const CudaSparseMatrix& other)
    : n_(other.n_), nnz_(other.nnz_) {
    cusparseHandle_t& cusparseHandle_ = CusparseHandle::getInstance();
    CHECK_CUDA(cudaMalloc((void**)&d_csrOffsets_, (n_ + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_cols_, nnz_ * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_vals_, nnz_ * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_csrOffsets_, other.d_csrOffsets_, (n_ + 1) * sizeof(int), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(d_cols_, other.d_cols_, nnz_ * sizeof(int), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(d_vals_, other.d_vals_, nnz_ * sizeof(float), cudaMemcpyDeviceToDevice));

    CHECK_CUSPARSE(cusparseCreateCsr(&matDescr_, n_, n_, nnz_,
        d_csrOffsets_, d_cols_, d_vals_,
        csr_row_ind_type_, csr_col_ind_type_,
        index_base_, valueType_));
}

CudaSparseMatrix::~CudaSparseMatrix() {
    CHECK_CUSPARSE(cusparseDestroySpMat(matDescr_));
    CHECK_CUDA(cudaFree(d_csrOffsets_));
    CHECK_CUDA(cudaFree(d_cols_));
    CHECK_CUDA(cudaFree(d_vals_));
}

void CudaSparseMatrix::updateData(const int* rows, const int* cols, const float* vals, int new_nnz, SparseType sparseType, MemoryType memType) {
    nnz_ = new_nnz;
    CHECK_CUDA(cudaFree(d_cols_));
    CHECK_CUDA(cudaFree(d_vals_));

    CHECK_CUDA(cudaMalloc((void**)&d_cols_, nnz_ * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_vals_, nnz_ * sizeof(float)));

    allocateAndCopy(rows, cols, vals, sparseType, memType);
}

CudaDenseVector CudaSparseMatrix::dot(const float* d_vec)
{
    float alpha = 1.0f;
    float beta = 0.0f;
    cusparseHandle_t& cusparseHandle_ = CusparseHandle::getInstance();
    CudaDenseVector result_vector = CudaDenseVector(n_);
    CudaDenseVector input_vector = CudaDenseVector(n_, d_vec, MemoryType::Device);
    
    size_t bufferSize = 0;
    void* dBuffer = nullptr;
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(cusparseHandle_,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matDescr_, input_vector.get(), &beta, result_vector.get(),
        valueType_, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));
    
    CHECK_CUSPARSE(cusparseSpMV(cusparseHandle_,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matDescr_, input_vector.get(), &beta, result_vector.get(),
        valueType_, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
    
    CHECK_CUDA(cudaFree(dBuffer));
    
    return result_vector;
}

void CudaSparseMatrix::allocateAndCopy(const int* rows, const int* cols, const float* vals, SparseType sparseType, MemoryType memType) {
    cudaMemcpyKind copyType;

    if (memType == MemoryType::Host) {
        copyType = cudaMemcpyHostToDevice;
        if (sparseType == SparseType::COO) {
            int* temp_d_rows;
            CHECK_CUDA(cudaMalloc((void**)&temp_d_rows, nnz_ * sizeof(int)));
            CHECK_CUDA(cudaMemcpy(temp_d_rows, rows, nnz_ * sizeof(int), copyType));
            rowsToCsr(temp_d_rows, d_csrOffsets_, n_, nnz_, sparseType);
            CHECK_CUDA(cudaFree(temp_d_rows));
        }
        else {
            CHECK_CUDA(cudaMemcpy(d_csrOffsets_, rows, (n_ + 1) * sizeof(int), copyType));
        }               
    }
    else {
        copyType = cudaMemcpyDeviceToDevice;
        if (sparseType == SparseType::COO) {
            rowsToCsr(rows, d_csrOffsets_, n_, nnz_, sparseType);
        }
        else {
            CHECK_CUDA(cudaMemcpy(d_csrOffsets_, rows, (n_ + 1) * sizeof(int), copyType));
        }
    }

    CHECK_CUDA(cudaMemcpy(d_cols_, cols, nnz_ * sizeof(int), copyType));
    CHECK_CUDA(cudaMemcpy(d_vals_, vals, nnz_ * sizeof(float), copyType));

}

void CudaSparseMatrix::rowsToCsr(const int* d_rows, int* d_csr_offset, int n, int nnz, SparseType sparseType)
{
    if (sparseType == SparseType::COO) {
        cusparseHandle_t& handle = CusparseHandle::getInstance();
        cusparseXcoo2csr(handle,
            d_rows,
            nnz,
            n,
            d_csr_offset,
            CUSPARSE_INDEX_BASE_ZERO);
    }
    
}

float* CudaSparseMatrix::sumRows()
{
    cusparseHandle_t& handle = CusparseHandle::getInstance();
    size_t bufferSize;
    void* dBuffer = nullptr;
    int* cscColPtr, *cscRowInd;
    float* cscVal, *diagonal;

    CHECK_CUDA(cudaMalloc((void**)&cscColPtr, (n_ + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&cscRowInd, nnz_ * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&cscVal, nnz_ * sizeof(float)));

    CHECK_CUDA(cudaMemset((void*)diagonal, 0, nnz_ * sizeof(float)));


    cusparseCsr2cscEx2_bufferSize(handle, n_, n_, nnz_,
        d_vals_, d_csrOffsets_, d_cols_,
        cscVal, cscColPtr, cscRowInd,
        CUDA_R_32F, CUSPARSE_ACTION_NUMERIC,
        CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1,
        &bufferSize);
    cudaMalloc(&dBuffer, bufferSize);

    // Conversion
    cusparseCsr2cscEx2(handle, n_, n_, nnz_,
        d_vals_, d_csrOffsets_, d_cols_,
        cscVal, cscColPtr, cscRowInd,
        CUDA_R_32F, CUSPARSE_ACTION_NUMERIC,
        CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1,
        dBuffer);

    // Clean up
    cudaFree(dBuffer);
    
    int blockSize = 512;
    int gridSize = (nnz_ + blockSize - 1) / blockSize;

    sum_axis << <gridSize, blockSize >> > (nnz_, cscRowInd, cscVal, diagonal);

    CHECK_CUDA(cudaFree(cscColPtr));
    CHECK_CUDA(cudaFree(cscRowInd));
    CHECK_CUDA(cudaFree(cscVal));

    return diagonal;
}

float* CudaSparseMatrix::sumCols()
{
    cusparseHandle_t& handle = CusparseHandle::getInstance();
    float* diagonal;

    CHECK_CUDA(cudaMemset((void*)diagonal, 0, nnz_ * sizeof(float)));

    int blockSize = 512;
    int gridSize = (nnz_ + blockSize - 1) / blockSize;

    sum_axis << <gridSize, blockSize >> > (nnz_, d_cols_, d_vals_, diagonal);

    return diagonal;
}

float* CudaSparseMatrix::sum(int axis)
{
    return nullptr;
}

void CudaSparseMatrix::display()
{
    cusparseHandle_t& handle = CusparseHandle::getInstance();
    float* d_denseMat;
    cudaMalloc((void**)&d_denseMat, n_ * n_ * sizeof(float));

    // Create a dense matrix descriptor
    cusparseDnMatDescr_t denseDescr;
    CHECK_CUSPARSE(cusparseCreateDnMat(&denseDescr,
        n_, // number of rows
        n_, // number of columns
        n_, // leading dimension
        d_denseMat, // pointer to dense matrix data
        CUDA_R_32F, // data type
        CUSPARSE_ORDER_ROW)); // row-major order

    // Convert sparse matrix to dense matrix
    void* dBuffer = NULL;
    size_t bufferSize = 0;
    CHECK_CUSPARSE(cusparseSparseToDense_bufferSize(handle,
        matDescr_,
        denseDescr,
        CUSPARSE_SPARSETODENSE_ALG_DEFAULT,
        &bufferSize));

    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

    CHECK_CUSPARSE(cusparseSparseToDense(handle,
        matDescr_,
        denseDescr,
        CUSPARSE_SPARSETODENSE_ALG_DEFAULT,
        dBuffer));

    // Copy the dense matrix from device to host
    std::vector<float> h_denseMat(n_ * n_);
    CHECK_CUDA(cudaMemcpy(h_denseMat.data(), d_denseMat, n_ * n_ * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << std::fixed << std::setprecision(4); // Set precision to 2 decimal places
    std::cout << "Dense matrix:" << std::endl;
    for (int i = 0; i < n_; ++i) {
        for (int j = 0; j < n_; ++j) {
            std::cout << h_denseMat[i * n_ + j] << " ";
        }
        std::cout << std::endl;
    }

    // Clean up
    CHECK_CUDA(cudaFree(d_denseMat));
    CHECK_CUDA(cudaFree(dBuffer));
    CHECK_CUSPARSE(cusparseDestroyDnMat(denseDescr));
}

int CudaSparseMatrix::getNnz() const
{
    return nnz_;
}

int CudaSparseMatrix::size() const
{
    return n_;
}

CudaDenseVector::CudaDenseVector(int size, const float* V, MemoryType memType): size_(size)
{
    cudaMemcpyKind copyType = memType == MemoryType::Host ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice;

    CHECK_CUDA(cudaMalloc((void**)&d_data_, size_ * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_data_, V, size_ * sizeof(int), copyType));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecDescr_, size_, d_data_, CUDA_R_32F));
}

CudaDenseVector::CudaDenseVector(int size)
{
    thrust::device_vector<float> input_vect = thrust::device_vector<float>(size_, 0.0f);
    CudaDenseVector(size_, thrust::raw_pointer_cast(input_vect.data()), MemoryType::Device);
}

CudaDenseVector::~CudaDenseVector()
{
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecDescr_));
    CHECK_CUDA(cudaFree(d_data_));
}

int CudaDenseVector::size() const
{
    return size_;
}

float* CudaDenseVector::data() const
{
    return d_data_;
}

cusparseDnVecDescr_t CudaDenseVector::get() const
{
    return vecDescr_;
}
