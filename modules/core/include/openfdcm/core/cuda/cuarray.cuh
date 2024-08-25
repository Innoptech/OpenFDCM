/*
MIT License

Copyright (c) 2024 Innoptech

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#ifndef OPENFDCM_CUDA_CUIMAGE_CUH
#define OPENFDCM_CUDA_CUIMAGE_CUH
#include "openfdcm/core/cuda/adapters.cuh"
#include "openfdcm/core/cuda/cumath.cuh"

#define OPENFDCM_TILE_DIM 32
#define OPENFDCM_BLOCK_ROWS 8

namespace openfdcm::core::cuda {

    static constexpr long Dynamic = -1;

    template <typename DerivedCuda, typename DerivedCpu>
    __host__ inline void copyToCuda(DerivedCuda& cudaArray, DerivedCpu const& cpuArray);

    template <typename DerivedCuda, typename DerivedCpu>
    __host__ inline void copyToCpu(DerivedCpu&& cpuArray, DerivedCuda const& cudaArray);

    template <typename DerivedCuda, typename DerivedCpu>
    __host__ inline void copyToCudaAsync(DerivedCuda& cudaArray, DerivedCpu const& cpuArray, const CudaStreamPtr &stream);

    template <typename DerivedCuda, typename DerivedCpu>
    __host__ inline void copyToCpuAsync(DerivedCpu& cpuArray, DerivedCuda const& cudaArray, const CudaStreamPtr &stream);

    /**
     * @brief Base class for all CUDA array-like structures using CRTP.
     *
     * This class provides a common interface for CudaArray-like structures.
     * It includes methods for accessing the underlying CUDA memory and the
     * size of the array. The derived classes should inherit from this class
     * to enable shared functionality.
     *
     * @tparam Derived The derived class (CRTP).
     */
    template<typename Derived>
    class CudaArrayBase {
    public:
        __host__ __device__
        Derived& derived() { return static_cast<Derived&>(*this); }

        __host__ __device__
        const Derived& derived() const { return static_cast<const Derived&>(*this); }

        // Common API to access array size
        __host__ __device__
        long size() const noexcept { return derived().size(); }

        // Accessor: Get value at (row, col)
        __host__ __device__
        auto& operator()(long row, long col) noexcept { return derived()(row, col); }

        __host__ __device__
        const auto& operator()(long row, long col) const noexcept { return derived()(row, col); }

        __host__ __device__
        auto* getArray() noexcept { return derived().getArray(); }
    };


    /**
     * @brief CUDA array class template supporting static and dynamic sizes.
     *
     * This class handles memory allocation and deallocation for arrays stored
     * on the GPU using CUDA. It supports both static and dynamic row/column
     * sizes. Inherits from CudaArrayBase to share a common interface.
     *
     * @tparam T The data type of the elements in the array.
     * @tparam Rows Number of rows (use core::cuda::Dynamic for dynamic size).
     * @tparam Cols Number of columns (use core::cuda::Dynamic for dynamic size).
     */
    template <typename T, long Rows, long Cols>
    class CudaArray : public CudaArrayBase<CudaArray<T, Rows, Cols>> {
        T* cuda_array_;         // Raw pointer for the device memory
        T* tmp_cuda_array_;     // Temporary storage for the device memory
        Eigen::Index cols_, rows_;       // Used for dynamic cases

        static T* allocateArray(long size) {
            T* cuda_array;
            cudaError_t cudaStatus = cudaMalloc((void**)&cuda_array, size * sizeof(T));
            if (cudaStatus != cudaSuccess) {
                throw std::runtime_error{ "cudaMalloc failed: " + std::string(cudaGetErrorString(cudaStatus)) };
            }
            return cuda_array;
        }

        void freeArray(T* array) {
            if (array) {
                cudaError_t cudaStatus = cudaFree(array);
                if (cudaStatus != cudaSuccess) {
                    std::cerr << "cudaFree failed: " << cudaGetErrorString(cudaStatus) << std::endl;
                }
            }
        }

        static constexpr long computeSize() noexcept {
            return (Rows == core::cuda::Dynamic || Cols == core::cuda::Dynamic) ? -1 : Rows * Cols;
        }

    public:
        using Scalar = T;
        static constexpr auto RowsAtCompileTime = Rows;
        static constexpr auto ColsAtCompileTime = Cols;
        static constexpr auto SizeAtCompileTime = Cols*Rows;

        // Ensure that Rows and Cols are not both zero
        static_assert(!(Rows == 0 && Cols == 0), "Rows and Cols cannot both be zero.");

        // Constructor for dynamic size
        CudaArray(long rows, long cols)
                : cuda_array_{ allocateArray(cols * rows) }
                , tmp_cuda_array_{ allocateArray(cols * rows) }
                , cols_{ cols }
                , rows_{ rows }
        {
            if constexpr (Rows != core::cuda::Dynamic){assert(Rows == rows);}
            if constexpr (Cols != core::cuda::Dynamic){assert(Cols == cols);}
        }

        // Constructor for when Rows == 1 or Cols == 1 (1D array)
        template <long R = Rows, long C = Cols,
                typename std::enable_if<(R == 1 || C == 1) && !(R == 0 && C == 0), long>::type = 0>
        CudaArray(long size)
                : cuda_array_{ allocateArray(size) }
                , tmp_cuda_array_{ allocateArray(size) }
                , cols_{ (Cols == 1) ? 1 : size }
                , rows_{ (Rows == 1) ? 1 : size }
        {
            static_assert((Rows == 1 || Cols == 1), "Constructor only available when Rows == 1 or Cols == 1.");
        }

        // Constructor for static size (compile-time known sizes)
        CudaArray()
                : cuda_array_{ allocateArray(computeSize()) }
                , tmp_cuda_array_{ allocateArray(computeSize()) }
                , cols_{ Cols }
                , rows_{ Rows }
        {
            static_assert(Rows != core::cuda::Dynamic && Cols != core::cuda::Dynamic, "Use default constructor only for static size.");
        }

        CudaArray(T const* data, long rows, long cols)
                : CudaArray{rows, cols}
        {
            cudaError_t cudaStatus = cudaMemcpy(reinterpret_cast<void*>(cuda_array_),
                                                reinterpret_cast<const void*>(data),
                                                rows * cols * sizeof(T),
                                                cudaMemcpyHostToDevice);
            if (cudaStatus != cudaSuccess) {
                throw std::runtime_error{"cudaMemcpy failed: " + std::string(cudaGetErrorString(cudaStatus))};
            }
        }

        template <long R = Rows, long C = Cols,
                typename std::enable_if<(R == 1 || C == 1) && !(R == 0 && C == 0), long>::type = 0>
        CudaArray(T const* data, long size)
                : cuda_array_{ allocateArray(size) }
                , tmp_cuda_array_{ allocateArray(size) }
                , cols_{ (Cols == 1) ? 1 : size }
                , rows_{ (Rows == 1) ? 1 : size }
        {
            static_assert((Rows == 1 || Cols == 1), "Constructor only available when Rows == 1 or Cols == 1.");

            cudaError_t cudaStatus = cudaMemcpy(reinterpret_cast<void*>(cuda_array_),
                                                reinterpret_cast<const void*>(data),
                                                size * sizeof(T),
                                                cudaMemcpyHostToDevice);
            if (cudaStatus != cudaSuccess) {
                throw std::runtime_error{"cudaMemcpy failed: " + std::string(cudaGetErrorString(cudaStatus))};
            }
        }

        ~CudaArray() {
            freeArray(cuda_array_);
            freeArray(tmp_cuda_array_);
        }

        // Disable copy semantics
        CudaArray(const CudaArray&) = delete;
        CudaArray& operator=(const CudaArray&) = delete;

        // Allow move semantics
        CudaArray(CudaArray&& other) noexcept
                : cuda_array_{ other.cuda_array_ }
                , tmp_cuda_array_{ other.tmp_cuda_array_ }
                , cols_{ other.cols_ }
                , rows_{ other.rows_ }
        {
            other.cuda_array_ = nullptr;
            other.tmp_cuda_array_ = nullptr;
        }

        CudaArray& operator=(CudaArray&& other) noexcept {
            if (this != &other) {
                freeArray(cuda_array_);
                freeArray(tmp_cuda_array_);

                cuda_array_ = other.cuda_array_;
                tmp_cuda_array_ = other.tmp_cuda_array_;
                cols_ = other.cols_;
                rows_ = other.rows_;

                other.cuda_array_ = nullptr;
                other.tmp_cuda_array_ = nullptr;
            }
            return *this;
        }

        // Size Accessors
        __host__ __device__ auto cols() const noexcept { return cols_; }
        __host__ __device__ auto rows() const noexcept { return rows_; }

        // Raw device array access
        __host__ __device__ T* getArray() const noexcept { return cuda_array_; }
        __host__ __device__ T* getTmpArray() const noexcept { return tmp_cuda_array_; }

        __device__
        T& operator()(long row, long col) noexcept { return cuda_array_[row + col * rows_]; }

        __device__
        const T& operator()(long row, long col) const noexcept { return cuda_array_[row + col * rows_]; }

        __device__
        T& operator()(long idx) noexcept requires (Rows == 1 || Cols == 1) {return cuda_array_[idx];}

        __device__
        const T& operator()(long idx) const noexcept requires (Rows == 1 || Cols == 1) {return cuda_array_[idx];}

        __host__ __device__ auto size() const noexcept { return cols_ * rows_; }

        __host__ void transpose(const CudaStreamPtr &stream);
    };

    template <typename DerivedArray>
    static constexpr bool isDynamic() noexcept {
        return (DerivedArray::RowsAtCompileTime == core::cuda::Dynamic || 
        DerivedArray::ColsAtCompileTime == core::cuda::Dynamic);
    }

    // Deduction guide for constructing CudaArray from Eigen matrix (Eigen::Matrix or Eigen::Array)
    template<typename Derived>
    CudaArray(Eigen::DenseBase<Derived> const& eigenCpuArray)
    -> CudaArray<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>;

    template<typename T, long Size> using cuVector = CudaArray<T, Size, 1>;
    using cuSize = cuVector<size_t, 2>;
    using cuPoint2 = cuVector<float, 2>;
    using cuMat22 = CudaArray<float, 2, 2>;
    using cuMat23 = CudaArray<float, 2, 3>;
    using cuLine = CudaArray<float, 4, 1>; // x1, y1, x2, y2
    using cuLineArray = CudaArray<float, 4, -1>;

    //---------------------------------------------------------------------------------------------------
    // Operations - Kernels
    //---------------------------------------------------------------------------------------------------
    template <typename T>
    __global__
    void transposeCoalesced(T *odata, const T *idata, int width, int height) {
        __shared__ float tile[OPENFDCM_TILE_DIM][OPENFDCM_TILE_DIM + 1];  // +1 to avoid bank conflicts

        int x = blockIdx.x * OPENFDCM_TILE_DIM + threadIdx.x;
        int y = blockIdx.y * OPENFDCM_TILE_DIM + threadIdx.y;

        // Load input tile into shared memory
        for (int j = 0; j < OPENFDCM_TILE_DIM; j += OPENFDCM_BLOCK_ROWS) {
            if ((x < width) && (y + j < height)) {
                tile[threadIdx.y + j][threadIdx.x] = idata[(y + j) * width + x];
            }
        }

        __syncthreads();

        // Transpose the block offset
        x = blockIdx.y * OPENFDCM_TILE_DIM + threadIdx.x;
        y = blockIdx.x * OPENFDCM_TILE_DIM + threadIdx.y;

        // Write the transposed tile to output
        for (int j = 0; j < OPENFDCM_TILE_DIM; j += OPENFDCM_BLOCK_ROWS) {
            if ((x < height) && (y + j < width)) {
                odata[(y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];
            }
        }
    }

    template <typename DerivedArray>
    __global__
    void sqrtKernel(DerivedArray& d_img) {
        // Calculate 2D index (x, y)
        int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
        int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

        // Ensure that the index is within bounds
        if (idx_x < d_img.cols() && idx_y < d_img.rows()) {
            d_img(idx_y, idx_x) = sqrtf(d_img(idx_y, idx_x));
        }
    }

    template <typename DerivedArray>
    __global__
    void setAllKernel(DerivedArray& d_img, typename DerivedArray::Scalar value) {
        // Calculate 2D index (x, y)
        int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
        int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

        // Ensure that the index is within bounds
        if (idx_x < d_img.cols() && idx_y < d_img.rows()) {
            d_img(idx_y, idx_x) = value;
        }
    }


    //---------------------------------------------------------------------------------------------------
    // Operations - Free functions
    //---------------------------------------------------------------------------------------------------
    template <typename DerivedArray>
    __host__
    inline void setAll(DerivedArray& array, typename DerivedArray::Scalar value, CudaStreamPtr const& stream) {
        dim3 threadsPerBlock(16, 16);  // Each block will have 16x16 threads
        dim3 numBlocks((array.cols() + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (array.rows() + threadsPerBlock.y - 1) / threadsPerBlock.y);
        setAllKernel<<<numBlocks, threadsPerBlock, 0, stream->getStream()>>>(array, value);
        synchronize(stream);
    }

    template <typename DerivedArray>
    __host__
    inline void sqrt(DerivedArray& array, CudaStreamPtr const& stream) {
        int threadsPerBlock = 256;
        int blocksPerGrid = (array.size() + threadsPerBlock - 1) / threadsPerBlock;
        sqrtKernel<<<blocksPerGrid, threadsPerBlock, 0, stream->getStream()>>>(array);
        synchronize(stream);
    }

    template <typename T>
    __host__
    inline void swap(T a, T b)
    {
        T c(a); a=b; b=c;
    }


    template <typename T, long Rows, long Cols>
    __host__
    inline void CudaArray<T,Rows,Cols>::transpose(CudaStreamPtr const& stream) {
        dim3 blockSize(OPENFDCM_TILE_DIM, OPENFDCM_BLOCK_ROWS);
        dim3 gridSize((cols_ + OPENFDCM_TILE_DIM - 1) / OPENFDCM_TILE_DIM, (rows_ + OPENFDCM_TILE_DIM - 1) / OPENFDCM_TILE_DIM);

        // Launch transpose kernel with the tmp array as output
        transposeCoalesced<<<gridSize, blockSize, 0, stream->getStream()>>>(tmp_cuda_array_, cuda_array_, cols_, rows_);
        synchronize(stream);

        // Swap the arrays and dimensions
        swap(cuda_array_, tmp_cuda_array_);
        swap(cols_, rows_);
    }

    template <typename DerivedArray>
    __host__
    inline void transpose(DerivedArray& array, CudaStreamPtr const& stream) {
        return array.transpose(stream);
    }

    template <typename DerivedCuda, typename DerivedCpu>
    __host__ inline void copyToCuda(DerivedCuda& cudaArray, DerivedCpu const& cpuArray)
    {
        static_assert(std::is_same_v<typename DerivedCuda::Scalar, typename DerivedCpu::Scalar>);
        if constexpr (!isDynamic<DerivedCuda> && !isDynamic<DerivedCpu>) {
            static_assert(DerivedCuda::Rows == DerivedCpu::Rows && DerivedCuda::Cols == DerivedCpu::Cols);
        } else {
            assert(cudaArray.rows() == cpuArray.rows() && cudaArray.cols() == cpuArray.cols());
        }

        // (void*) is absolutely necessary
        cudaError_t cudaStatus = cudaMemcpy(reinterpret_cast<void*>(cudaArray.getArray()),
                                            reinterpret_cast<const void*>(cpuArray.derived().data()),
                                            cpuArray.size() * sizeof(typename DerivedCpu::Scalar),
                                            cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            throw std::runtime_error{"cudaMemcpy failed: " + std::string(cudaGetErrorString(cudaStatus))};
        }
    }

    template <typename DerivedCuda, typename DerivedCpu>
    __host__ inline void copyToCpu(DerivedCpu&& cpuArray, DerivedCuda const& cudaArray)
    {
        static_assert(std::is_same_v<typename DerivedCuda::Scalar, typename DerivedCpu::Scalar>);
        if constexpr (!isDynamic<DerivedCuda> && !isDynamic<DerivedCpu>()) {
            static_assert(DerivedCuda::Rows == DerivedCpu::Rows && DerivedCuda::Cols == DerivedCpu::Cols);
        } else {
            assert(cudaArray.rows() == cpuArray.rows() && cudaArray.cols() == cpuArray.cols());
        }

        cudaError_t cudaStatus = cudaMemcpy(reinterpret_cast<void*>(cpuArray.derived().data()),
                                            reinterpret_cast<const void*>(cudaArray.getArray()),
                                            cudaArray.size() *  sizeof(typename DerivedCpu::Scalar),
                                            cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            throw std::runtime_error{"cudaMemcpy failed: " + std::string(cudaGetErrorString(cudaStatus))};
        }
    }

    template <typename DerivedCuda, typename DerivedCpu>
    __host__ inline void copyToCudaAsync(DerivedCuda& cudaArray, DerivedCpu const& cpuArray, const CudaStreamPtr &stream)
    {
        static_assert(std::is_same_v<typename DerivedCuda::Scalar, typename DerivedCpu::Scalar>);
        if constexpr (!isDynamic<DerivedCuda> && !isDynamic<DerivedCpu>) {
            static_assert(DerivedCuda::Rows == DerivedCpu::Rows && DerivedCuda::Cols == DerivedCpu::Cols);
        } else {
            assert(cudaArray.rows() == cpuArray.rows() && cudaArray.cols() == cpuArray.cols());
        }
        cudaError_t cudaStatus = cudaMemcpyAsync(reinterpret_cast<void*>(cudaArray.getArray()),
                                                 reinterpret_cast<const void*>(cpuArray.derived().data()),
                                                 cpuArray.size() * sizeof(typename DerivedCpu::Scalar),
                                                 cudaMemcpyHostToDevice, stream->getStream());
        if (cudaStatus != cudaSuccess) {
            throw std::runtime_error{"cudaMemcpyAsync failed: " + std::string(cudaGetErrorString(cudaStatus))};
        }
    }

    template <typename DerivedCuda, typename DerivedCpu>
    __host__
    inline void copyToCpuAsync(DerivedCpu& cpuArray, DerivedCuda const& cudaArray, const CudaStreamPtr &stream)
    {
        static_assert(std::is_same_v<typename DerivedCuda::Scalar, typename DerivedCpu::Scalar>);
        if constexpr (!isDynamic<DerivedCuda> && !isDynamic<DerivedCpu>) {
            static_assert(DerivedCuda::Rows == DerivedCpu::Rows && DerivedCuda::Cols == DerivedCpu::Cols);
        } else {
            assert(cudaArray.rows() == cpuArray.rows() && cudaArray.cols() == cpuArray.cols());
        }
        cudaError_t cudaStatus = cudaMemcpyAsync(reinterpret_cast<void*>(cpuArray.derived().data()),
                                                 reinterpret_cast<const void*>(cudaArray.getArray()),
                                                 cudaArray.size() *  sizeof(typename DerivedCpu::Scalar),
                                                 cudaMemcpyDeviceToHost, stream->getStream());
        if (cudaStatus != cudaSuccess) {
            throw std::runtime_error{"cudaMemcpyAsync failed: " + std::string(cudaGetErrorString(cudaStatus))};
        }
    }

} //namespace openfdcm::cuda
#endif //OPENFDCM_CUDA_CUIMAGE_CUH
