#include "CudaSparseMatrix.hpp"

#include <iostream>
#include <iomanip>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>


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
    : n_(other.n_), nnz_(other.nnz_), matDescr_(nullptr) {
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
    clear();
}

void CudaSparseMatrix::updateData(const int* rows, const int* cols, const float* vals, int new_nnz, SparseType sparseType, MemoryType memType) {
    nnz_ = new_nnz;
    CHECK_CUDA(cudaFree(d_cols_));
    CHECK_CUDA(cudaFree(d_vals_));
    CHECK_CUSPARSE(cusparseDestroySpMat(matDescr_));

    CHECK_CUDA(cudaMalloc((void**)&d_cols_, nnz_ * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_vals_, nnz_ * sizeof(float)));

    allocateAndCopy(rows, cols, vals, sparseType, memType);

    CHECK_CUSPARSE(cusparseCreateCsr(&matDescr_, n_, n_, nnz_,
        d_csrOffsets_, d_cols_, d_vals_,
        csr_row_ind_type_, csr_col_ind_type_,
        index_base_, valueType_));
}

bool* CudaSparseMatrix::zero_elements_in_vector(const float* input_vect, int& zero_sum, int n) {
    cusparseHandle_t& handle = CusparseHandle::getInstance();
    bool* zero_elements_vect;
    int* d_zero_sum;

    zero_sum = 0;

    // Allocate memory on the device
    CHECK_CUDA(cudaMalloc((void**)&zero_elements_vect, n * sizeof(bool)));
    CHECK_CUDA(cudaMalloc((void**)&d_zero_sum, sizeof(int)));

    // Initialize memory
    CHECK_CUDA(cudaMemset(zero_elements_vect, 0, n * sizeof(bool)));
    CHECK_CUDA(cudaMemset(d_zero_sum, 0, sizeof(int)));

    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    zero_elements << <gridSize, BLOCK_SIZE >> > (input_vect, zero_elements_vect, d_zero_sum, n);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy the result back to host
    CHECK_CUDA(cudaMemcpy(&zero_sum, d_zero_sum, sizeof(int), cudaMemcpyDeviceToHost));

    // Free device memory
    CHECK_CUDA(cudaFree(d_zero_sum));

    return zero_elements_vect;
}

void CudaSparseMatrix::fill_diagonal(const float* diagonal_vect)
{
    int nnz_sum = 0;
    int zero_sum = 0;
    int diag_nnz = n_;
    int resize_n = diag_nnz;
    // TODO: Flip to non zero vector and copy only the non zero elements to new vector
    bool* zeros_in_diag_sum = zero_elements_in_vector(diagonal_vect, zero_sum, n_);
    bool* non_zero = non_zero_diagonal(nnz_sum);
    
    bool* h_non_zero = new bool[n_];
    CHECK_CUDA(cudaMemcpy(h_non_zero, zeros_in_diag_sum, n_ * sizeof(bool), cudaMemcpyDeviceToHost));

    for (int i = 0; i < n_; i++)
    {
        std::cout << "zero_in_diag_" << i << ": " << h_non_zero[i] << std::endl;
    }
    
    diag_nnz -= zero_sum;
    resize_n = diag_nnz - nnz_sum;

    int* original_I, * new_I, * new_J;
    float* new_V;

    CHECK_CUDA(cudaMalloc((void**)&original_I, (nnz_) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&new_I, (nnz_ + resize_n) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&new_J, (nnz_ + resize_n) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&new_V, (nnz_ + resize_n) * sizeof(float)));

    csrTorows(d_csrOffsets_, original_I, n_, nnz_, SparseType::CSR);

    CHECK_CUDA(cudaMemcpy(new_I, original_I, nnz_ * sizeof(int), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(new_J, d_cols_, nnz_ * sizeof(int), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(new_V, d_vals_, nnz_ * sizeof(float), cudaMemcpyDeviceToDevice));

    int gridSize = ((nnz_+n_)+BLOCK_SIZE - 1) / BLOCK_SIZE;
    set_diagonal << <gridSize, BLOCK_SIZE >> > (new_I, new_J, new_V, non_zero, diagonal_vect, nnz_, resize_n);
    CHECK_CUDA(cudaDeviceSynchronize());
    thrust::device_ptr<int> dev_I(new_I);
    thrust::device_ptr<int> dev_J(new_J);
    thrust::device_ptr<float> dev_V(new_V);

    // First, sort by the secondary key (J) using stable sort
    thrust::stable_sort_by_key(dev_J, dev_J + (nnz_ + resize_n), thrust::make_zip_iterator(thrust::make_tuple(dev_I, dev_V)));

    // Then, sort by the primary key (I) using stable sort to maintain the order of the secondary key
    thrust::stable_sort_by_key(dev_I, dev_I + (nnz_ + resize_n), thrust::make_zip_iterator(thrust::make_tuple(dev_J, dev_V)));

    float* h_new_I = new float[nnz_+resize_n];
    CHECK_CUDA(cudaMemcpy(h_new_I, new_V, (nnz_ + resize_n) * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < nnz_ + resize_n; i++)
    {
        std::cout << "Copied V_" << i << ": " << h_new_I[i] << std::endl;
    }

    updateData(new_I, new_J, new_V, nnz_ + resize_n, SparseType::COO, MemoryType::Device);

    CHECK_CUDA(cudaFree(original_I));
    CHECK_CUDA(cudaFree(new_I));
    CHECK_CUDA(cudaFree(new_J));
    CHECK_CUDA(cudaFree(new_V));

    std::cout << "Total non zero elements on the diagonal: " << nnz_sum << std::endl;
}

bool* CudaSparseMatrix::non_zero_diagonal(int& nnz_diag_sum)
{
    cusparseHandle_t& handle = CusparseHandle::getInstance();
    bool* nnz_diag;
    int* I;
    int* d_nnz_diag_sum;
    nnz_diag_sum = 0;

    // Allocate memory on the device
    CHECK_CUDA(cudaMalloc((void**)&d_nnz_diag_sum, sizeof(int)));

    // Initialize memory
    CHECK_CUDA(cudaMemset(d_nnz_diag_sum, 0, sizeof(int)));

    CHECK_CUDA(cudaMalloc((void**)&I, nnz_ * sizeof(int)));
    cusparseXcsr2coo(handle,
        d_csrOffsets_,
        nnz_,
        n_,
        I,
        CUSPARSE_INDEX_BASE_ZERO);


    CHECK_CUDA(cudaMalloc((void**)&nnz_diag, n_ * sizeof(bool)));
    CHECK_CUDA(cudaMemset(nnz_diag, 0, n_));
    int gridSize = (nnz_ + BLOCK_SIZE - 1) / BLOCK_SIZE;
    non_zero_elements << <gridSize, BLOCK_SIZE >> > (I, d_cols_, nnz_diag, d_nnz_diag_sum, nnz_);

    CHECK_CUDA(cudaMemcpy(&nnz_diag_sum, d_nnz_diag_sum, sizeof(int), cudaMemcpyDeviceToHost));

    // Free device memory
    CHECK_CUDA(cudaFree(d_nnz_diag_sum));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaFree(I));

    return nnz_diag;
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

void CudaSparseMatrix::multiply(float value)
{
    cusparseHandle_t& handle = CusparseHandle::getInstance();
    // Set scaling factors
    const float beta = 0.0f;
    size_t bufferSize = 0;

    cusparseMatDescr_t input_desc;
    CHECK_CUSPARSE(cusparseCreateMatDescr(&input_desc));

    // Create matrix descriptor for the result matrix C
    cusparseMatDescr_t result_desc;
    CHECK_CUSPARSE(cusparseCreateMatDescr(&result_desc));

    // Get buffer size for the operation
    CHECK_CUSPARSE(cusparseScsrgeam2_bufferSizeExt(handle,
        n_, n_,
        &value, input_desc, nnz_,
        d_vals_,
        d_csrOffsets_,
        d_cols_,
        &beta, nullptr, nnz_,
        nullptr,
        nullptr,
        nullptr,
        result_desc,
        d_vals_,
        d_csrOffsets_,
        d_cols_,
        &bufferSize));

    void* dBuffer;
    cudaMalloc(&dBuffer, bufferSize);

    // Perform the scaling operation
    cusparseScsrgeam2(handle,
        n_, n_,
        &value, input_desc, nnz_,
        d_vals_,
        d_csrOffsets_,
        d_cols_,
        &beta, input_desc, nnz_,
        d_vals_,
        d_csrOffsets_,
        d_cols_,
        result_desc,
        d_vals_,
        d_csrOffsets_,
        d_cols_,
        dBuffer);
    // Clean up
    cudaFree(dBuffer);
    cusparseDestroyMatDescr(input_desc);
    cusparseDestroyMatDescr(result_desc);

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

void CudaSparseMatrix::csrTorows(const int* d_csr_offset, int* d_rows, int n, int nnz, SparseType sparseType)
{
    if (sparseType == SparseType::CSR) {
        cusparseHandle_t& handle = CusparseHandle::getInstance();
        cusparseXcsr2coo(handle,
            d_csr_offset,
            nnz,
            n,
            d_rows,
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

    CHECK_CUDA(cudaMalloc((void**)&diagonal, n_ * sizeof(float)));
    CHECK_CUDA(cudaMemset((void*)diagonal, 0, n_ * sizeof(float)));


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
    
    int gridSize = (nnz_ + BLOCK_SIZE - 1) / BLOCK_SIZE;

    sum_axis << <gridSize, BLOCK_SIZE >> > (nnz_, cscRowInd, cscVal, diagonal);

    CHECK_CUDA(cudaFree(cscColPtr));
    CHECK_CUDA(cudaFree(cscRowInd));
    CHECK_CUDA(cudaFree(cscVal));

    return diagonal;
}

float* CudaSparseMatrix::sumCols()
{
    cusparseHandle_t& handle = CusparseHandle::getInstance();
    float* diagonal;

    CHECK_CUDA(cudaMalloc((void**)&diagonal, n_ * sizeof(float)));
    CHECK_CUDA(cudaMemset((void*)diagonal, 0, n_ * sizeof(float)));

    int blockSize = 512;
    int gridSize = (nnz_ + blockSize - 1) / blockSize;

    sum_axis << <gridSize, blockSize >> > (nnz_, d_cols_, d_vals_, diagonal);

    return diagonal;
}

float* CudaSparseMatrix::sum(int axis)
{
    if (axis == 0) {
        return sumRows();
    }

    if (axis == 1) {
        return sumCols();
    }

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
            std::cout << std::setw(7) << h_denseMat[i * n_ + j] << " ";
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

void CudaSparseMatrix::clear()
{
    if (d_csrOffsets_) {
        CHECK_CUDA(cudaFree(d_csrOffsets_));
        std::cout << "d_csrOffsets_ cleared" << std::endl;
        d_csrOffsets_ = nullptr;
    }
    if (d_cols_) {
        CHECK_CUDA(cudaFree(d_cols_));
        std::cout << "d_cols_ cleared" << std::endl;
        d_cols_ = nullptr;
    }
    if (d_vals_) {
        CHECK_CUDA(cudaFree(d_vals_));
        std::cout << "d_vals_ cleared" << std::endl;
        d_vals_ = nullptr;
    }
    if (matDescr_) {
        CHECK_CUSPARSE(cusparseDestroySpMat(matDescr_));
        std::cout << "matDescr_ cleared" << std::endl;
        matDescr_ = nullptr;
    }

    nnz_ = 0;
    n_ = 0;
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
