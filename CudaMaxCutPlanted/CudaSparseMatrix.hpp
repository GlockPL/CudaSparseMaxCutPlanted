#ifndef CUDA_SPARSE_MATRIX_HPP
#define CUDA_SPARSE_MATRIX_HPP

#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <memory>
#include "Kernels.cuh"


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

enum class MemoryType {
    Host,
    Device
};


enum class SparseType {
    COO,
    CSR
};

class CusparseHandle {
public:
    static cusparseHandle_t& getInstance() {
        static cusparseHandle_t handle;
        static bool initialized = false;
        if (!initialized) {
            cusparseCreate(&handle);
            initialized = true;
        }
        return handle;
    }
    CusparseHandle(const CusparseHandle&) = delete;
    CusparseHandle& operator=(const CusparseHandle&) = delete;
private:
    CusparseHandle() {}
    ~CusparseHandle() {
        cusparseDestroy(getInstance());
    }
};

class CudaDenseVector {
public:
    CudaDenseVector(int size, const float* V, MemoryType memType);
    CudaDenseVector(int size);
    ~CudaDenseVector();

    int size() const;
    float* data() const;
    cusparseDnVecDescr_t get() const;

private:
    int size_;
    float* d_data_;
    cusparseDnVecDescr_t vecDescr_;
};

class CudaSparseMatrix {
public:
    CudaSparseMatrix(int* I, int* J, float* V, int n, int nnz, SparseType sparseType, MemoryType memType);
    CudaSparseMatrix(const CudaSparseMatrix& other);
    ~CudaSparseMatrix();

    void updateData(const int* rows, const int* cols, const float* vals, int new_nnz, SparseType sparseType, MemoryType memType);
    void fill_diagonal(const float* diagonal_vect);
    bool* non_zero_diagonal(int& nnz_diag_sum);
    CudaDenseVector dot(const float* d_vec);
    void multiply(float value);
    float* sum(int axis);
    void display();
    int getNnz() const;
    int size() const;
    void clear();

    // Other useful methods can be added here

private:
    int n_;
    int nnz_;
    int* d_csrOffsets_;
    int* d_cols_;
    float* d_vals_;
    cusparseSpMatDescr_t matDescr_;
    cusparseIndexType_t csr_row_ind_type_ = CUSPARSE_INDEX_32I;
    cusparseIndexType_t csr_col_ind_type_ = CUSPARSE_INDEX_32I;
    cusparseIndexBase_t index_base_ = CUSPARSE_INDEX_BASE_ZERO;
    cudaDataType valueType_ = CUDA_R_32F;
    void allocateAndCopy(const int* rows, const int* cols, const float* vals, SparseType sparseType, MemoryType memType);
    void rowsToCsr(const int* d_rows, int* d_csr_offset, int n, int nnz, SparseType sparseType);
    void csrTorows(const int* d_csr_offset, int* d_rows, int n, int nnz, SparseType sparseType);
    bool* zero_elements_in_vector(const float* input_vect, int& zero_sum, int n);
    float* sumRows();
    float* sumCols();

};

#endif // CUDA_SPARSE_MATRIX_HPP
